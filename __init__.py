"""ComfyUI-Krea2Edit — in-context edit forward for the Krea2 model.

ComfyUI's native Krea2 `_forward` is text-to-image only: it builds the sequence
`[text | target]`. The krea2_edit LoRA (trained in ai-toolkit) needs the *appearance
path*: the VAE-encoded SOURCE latent prepended as a block of clean tokens, distinguished
from the (noisy) target purely by the 3-axis RoPE frame index (source=1, target=0, h/w
aligned). This node adds that by wrapping the model's DIFFUSION_MODEL forward and rebuilding
the sequence as `[text | source(frame=1) | target(frame=0)]`, keeping only the target tokens
out — mirroring ai-toolkit's `predict_velocity_edit` exactly, using the model's own submodules.

Wiring:  LoadImage -> VAEEncode(source) --\
                                            Krea2EditAdvanced(model, source_latent_a) -> KSampler
         UNETLoader -> LoraLoaderModelOnly -/
KSampler.latent_image <- EmptySD3LatentImage (noise). Text: NATIVE krea2 CLIP + CLIPTextEncode.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.patcher_extension
import comfy.samplers
import comfy.utils
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy_api.latest import ComfyExtension, io


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


REF_POSITIONS = ["aligned (trained)", "beside (right of target)", "bag of features"]


def _place_refs(ref_ids, modes, h_, w_):
    """Reference placement on the h/w RoPE axes (frame axis untouched: refs keep 1..N).
    aligned (trained): ref token (h, w) sits on target token (h, w) -- how the models were trained.
    beside: each ref is shifted to the right of the target canvas (and of earlier refs): no h/w overlap,
            vertical placement and the ref's internal layout unchanged.
    bag of features (experimental): every token of a ref gets one position (the target's centre), so
            attention to it is decided by content alone; the ref's layout is discarded.
    modes: one per ref, e.g. scene aligned + subject beside."""
    out, x0 = [], float(w_)
    for ids, mode in zip(ref_ids, modes):
        if mode.startswith("aligned"):
            out.append(ids)
            continue
        ids = ids.clone()
        if mode.startswith("beside"):
            ids[..., 2] = ids[..., 2] - ids[..., 2].min() + x0
            x0 = float(ids[..., 2].max()) + 1.0
        elif mode.startswith("bag"):
            ids[..., 1] = (h_ - 1) / 2.0
            ids[..., 2] = (w_ - 1) / 2.0
        else:
            raise ValueError(f"unknown ref_position {mode!r}")
        out.append(ids)
    return out


def _to_4d(v):
    """(B,C,T,H,W) -> (B*T,C,H,W); pass 4D through. Images use T=1."""
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _fit_box(ih, iw, px_h, px_w):
    """Window (y0, x0, h, w) of an ih x iw source that is cropped, centered, to the aspect ratio of a px_h x px_w target.
    Source latents and their boost masks follow this one geometry."""
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    return (ih - ch) // 2, (iw - cw) // 2, ch, cw


def _crop_mask(mask, ih, iw, box):
    """Crop a MASK drawn on an ih x iw source to `box` (y0, x0, h, w) of that source. The mask may
    have a different resolution than the source it was drawn on."""
    y0, x0, ch, cw = box
    mh, mw = mask.shape[-2:]
    ya, xa = round(y0 * mh / ih), round(x0 * mw / iw)
    yb, xb = max(ya + 1, round((y0 + ch) * mh / ih)), max(xa + 1, round((x0 + cw) * mw / iw))
    return mask[..., ya:yb, xa:xb]


def _fit_src(src, H, W):
    """Fit a source latent to the target grid the way TRAINING did: center-crop to
    the target aspect ratio, then resize. A plain interpolate (the pre-fix behavior)
    STRETCHES mixed-AR sources — users saw stretched people whenever their input AR
    differed from the output resolution."""
    sh, sw = src.shape[-2:]
    if (sh, sw) == (H, W):
        return src
    y0, x0, ch, cw = _fit_box(sh, sw, H, W)
    src = src[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(src.float(), size=(H, W), mode="bilinear")


def _place_ref(ref, mask, H, W):
    """Reference a on the target's latent grid: padded to whole tokens and fitted like krea2_edit_forward fits a
    source. Returns the reference, its coverage and its mask weight (the MASK drawn on the reference, 1 without one),
    cropped to H x W."""
    Hp, Wp = -(-H // TOKEN) * TOKEN, -(-W // TOKEN) * TOKEN
    ref = comfy.ldm.common_dit.pad_to_patch_size(_to_4d(ref), (TOKEN, TOKEN), padding_mode="replicate")
    if ref.shape[-2:] != (Hp, Wp):
        if mask is not None:
            mask = _crop_mask(mask, *ref.shape[-2:], _fit_box(*ref.shape[-2:], Hp, Wp))
        ref = _fit_src(ref, Hp, Wp)
    weight = ref.new_ones(1, 1, Hp, Wp)
    if mask is not None:
        weight = F.interpolate((mask if mask.ndim == 3 else mask[None])[:1][None].float(), size=(Hp, Wp), mode="area")
    return ref[..., :H, :W], ref.new_ones(1, 1, H, W), weight[..., :H, :W]


def _ref_attn_bias(boosts, masks, txtlen, slens, tgtlen, mask_hw, device, dtype):
    """Additive attention-logit bias on the [text | refs... | target] sequence.

    boosts: per-ref factor on target->ref attention, aligned with the source blocks.
    Equivalent to multiplying those keys' post-softmax attention weight before
    renormalization. masks: per-ref ComfyUI MASK (or None), already cropped to what the
    reference shows, scaling that ref's boost per token (e.g. the face): a token whose averaged
    mask value is m gets the attention weight 1 + m * (boost - 1).
    """
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, sl = offs[i], slens[i]
        if masks[i] is None:
            bias[:, :, rows0:, off:off + sl] = math.log(max(b, 1e-4))
            continue
        mask = masks[i] if masks[i].ndim == 3 else masks[i][None]
        mask = F.interpolate(mask[:1][None].float(), size=mask_hw[i], mode="area").reshape(-1).to(device)
        bias[:, :, rows0:, off:off + sl] = torch.log((1 + mask * (b - 1)).clamp_min(1e-4))
    return bias


def krea2_edit_forward(m, x, timesteps, context, src_latent, transformer_options,
                       ref_boost_a=1.0, ref_boost_b=1.0, ref_boost_mask_a=None, ref_boost_mask_b=None,
                       ref_position_a="aligned (trained)", ref_position_b="aligned (trained)"):
    """Krea2 SingleStreamDiT._forward, but with source block(s) prepended.

    m           : the SingleStreamDiT (LoRA-patched at sample time)
    x           : (B,C,H,W) or (B,C,T,H,W) noisy TARGET latent
    src_latent  : clean SOURCE latent (VAE-encoded), 4D/5D — or a LIST of them
                  (multi-ref: [a, b] = [scene, subject], frames 1..N, training-matched).
                  The *_a controls always act on the first entry, the *_b controls on the second.
    context     : (B, seq, txtlayers*txtdim) — the 12-layer Qwen3-VL stack
    """
    patch = m.patch

    # Mirror ComfyUI _forward: latents may arrive 5D (B,C,T,H,W) for this model.
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    # source(s) -> (bs, C, H, W): flatten temporal, match batch, fit to the target grid
    # (center-crop to target AR then resize — training-matched; never stretch).
    src_list = src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
    masks = [ref_boost_mask_a, ref_boost_mask_b]
    srcs = []
    for i, sl in enumerate(src_list):
        src = _to_4d(sl).to(x.device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        if src.shape[-2:] != (H, W):
            print(f"[krea2edit] LATENT-PATH fit_src (crop): src={tuple(src.shape[-2:])} -> {H}x{W}", flush=True)
            sh, sw = src.shape[-2:]
            if masks[i] is not None:
                masks[i] = _crop_mask(masks[i], sh, sw, _fit_box(sh, sw, H, W))
            src = _fit_src(src, H, W).to(x.dtype)
        srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch), padding_mode="replicate"))
    src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

    context = m._unpack_context(context)                       # (B, seq, 12, 2560)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    src_imgs = [m.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                for s_ in srcs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    srclen = sum(si.shape[1] for si in src_imgs)
    combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)  # [text | refs... | target]

    device = combined.device
    ref_ids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(src_grids)]
    ref_ids = _place_refs(ref_ids, [ref_position_a, ref_position_b][:len(ref_ids)], h_, w_)
    pos = torch.cat([
        torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]   # text @ 0
        + ref_ids
        + [_imgids(bs, 0, h_, w_, device)],                                    # target frame=0
        dim=1)
    freqs = m.pe_embedder(pos)

    attn_bias = None
    boosts = [ref_boost_a, ref_boost_b][:len(src_imgs)]
    if any(b != 1.0 for b in boosts):
        attn_bias = _ref_attn_bias(boosts, masks[:len(src_imgs)], txtlen,
                                   [si.shape[1] for si in src_imgs], tgtlen,
                                   src_grids, combined.device, combined.dtype)

    for block in m.blocks:
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]         # target tokens only
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


# ---------------------------------------------------------------------------------------------------------
# Reference guidance: pull the denoised estimate toward the aligned reference latent.
#
# At every step the sampler's x0 estimate is blended toward the reference, x0' = x0 + w * (ref - x0). This is
# the same update as blending the velocity toward the reference's straight-line velocity (RF-Inversion's
# controller guidance). Output preservation is decided by the late steps, layout by the early ones, so the
# schedule has two phases split at `crossover * sigma_start`. Two optional gates let the model deviate where
# it clearly wants to:
# - distance gate (late phase): tokens whose estimate is far from the reference are left alone;
# - edit-map gate: tokens where the instruction changes the prediction (cond vs uncond) are left alone.
# ---------------------------------------------------------------------------------------------------------
TOKEN = 2  # latent cells per token side (patch size)


@dataclass
class GuideConfig:
    ref_guide: float = 0.0     # pull strength 0..1; 0 = off, 1 = copy the reference where the mask allows
    crossover: float = 0.5     # early phase = sigma above crossover * sigma_start
    late_gain: float = 1.0     # late-phase strength = min(1, late_gain * ref_guide); 0 = early phase only
    early_gain: float = 1.0    # early-phase strength = early_gain * ref_guide; 0 = late phase only
    early_decay: float = 0.0   # LAB: the early pull falls with sigma as ((sigma - crossover * sigma0) / (sigma0 - crossover * sigma0)) ** early_decay; 0 = constant
    early_hold: float = 0.0    # LAB: with early_outside, the early pull where the mask is 0 (everywhere without a mask) peaks at min(X, 1) and falls with sigma with exponent 2 / X; 0 = off
    dist_tau: float = 0.0      # distance-gate tolerance in reference-std units; 0 = off
    dist_calib: float = 0.0    # LAB: tolerance = dist_calib * the 90th percentile of the distance outside the mask; 0 = use dist_tau
    dist_pow: float = 2.0      # LAB: steepness n of the gate d^n / (d^n + tol^n)
    dist_blur: float = 0.0     # LAB: blur of the distance map, in tokens
    free_latch: bool = False   # LAB: a token the gate has freed stays free for the rest of the late phase
    free_grow: int = 0         # LAB: the freed region grows by this many tokens (then softens by one token)
    edit_thr: float = 0.0      # edit-map threshold in (0, 1); 0 = off
    edit_start: float = 0.9    # edit map is estimated for sigma in [edit_end, edit_start] * sigma_start
    edit_end: float = 0.5
    edit_soft: float = 0.3     # LAB: width of the smoothstep around the edit threshold
    edit_q: float = 0.95       # LAB: the edit map is normalised by this quantile of its values
    edit_grow: int = 0         # LAB: the region the edit map frees grows by this many tokens
    edit_region_norm: bool = False  # LAB: the edit map is normalised over the masked region (mask < 0.5) instead of the whole image
    gate_and: bool = False     # LAB: a token is freed only when the distance gate AND the edit gate agree (default: either)
    gate_gamma: float = 1.0    # LAB: with gate_in_mask the freedom is scaled by (1 - mask) ** gate_gamma
    early_outside: bool = False  # LAB: the early-phase pull ignores the inside floor, i.e. is zero where the mask is 0
    inside: float = 0.0       # pull where the mask is 0, as a fraction of the pull where it is 1 (linear in between)
    gate_in_mask: bool = False  # the gates free tokens only where the mask is 0; where it is 1 the pull is unconditional
    feather: float = 0.0       # blur radius of the weight map, as a fraction of the latent side
    smooth: float = 0.5        # across-step memory of the gate maps (0 = none)
    debug: bool = False        # LAB ONLY: print gate statistics and dump the maps as PNGs under output/guide_lab_debug


def step_weights(sigmas, ref_guide, crossover, late_gain, early_gain=1.0, early_decay=0.0):
    """Per step (early, late) weights. A step that straddles the crossover is split by its share of the
    pull (1 - sigma_next / sigma), so the total early pull does not depend on the step count."""
    sigmas = [float(s) for s in sigmas]
    sigma_c = crossover * sigmas[0]
    late = min(1.0, late_gain * ref_guide)
    plan = []
    for s, sn in zip(sigmas[:-1], sigmas[1:]):
        f = min(1.0, max(0.0, (1 - sigma_c / s) / (1 - sn / s))) if s > sigma_c else 0.0
        fall = ((s - sigma_c) / (sigmas[0] - sigma_c)) ** early_decay if s > sigma_c and sigmas[0] > sigma_c else 1.0
        plan.append((early_gain * ref_guide * f * fall, late * (1 - f)))
    return plan


def token_rms(a, b):
    """RMS over channels of a - b, averaged over each token, back at latent-cell resolution."""
    sq = (a - b).float().pow(2).mean(1, keepdim=True)
    sq = F.avg_pool2d(sq, TOKEN, ceil_mode=True)
    return F.interpolate(sq, size=a.shape[-2:], mode="nearest").sqrt()


def blur(m, radius):
    if radius < 1:
        return m
    return F.avg_pool2d(F.pad(m, (radius,) * 4, mode="replicate"), 2 * radius + 1, stride=1)


def _grow(m, tokens):
    """Grow a map by `tokens` tokens (max filter), then soften it by one token."""
    r = tokens * TOKEN
    return blur(F.max_pool2d(m, 2 * r + 1, stride=1, padding=r), TOKEN)


def _smoothstep(x):
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


class Guide:
    """State of one sampling run. Build a new one per run."""

    def __init__(self, cfg, sigmas):
        self.cfg = cfg
        self.sigmas = [float(s) for s in sigmas]
        self.plan = step_weights(self.sigmas, cfg.ref_guide, cfg.crossover, cfg.late_gain, cfg.early_gain, cfg.early_decay)
        self.hold = [e for e, _ in step_weights(self.sigmas, cfg.ref_guide, cfg.crossover, cfg.late_gain, min(cfg.early_hold, 1.0), 2.0 / cfg.early_hold)] if cfg.early_hold > 0 else [0.0] * len(self.plan)
        self.ref_std = None
        self.dist = None
        self.free_max = None
        self.edit_ema = None
        self.edit_acc = None
        self.tag = None

    def _debug(self, i, sigma, w):
        """LAB ONLY (remove with the lab node): per-step statistics and maps."""
        import os
        import time
        from PIL import Image
        self.tag = self.tag or time.strftime("%H%M%S")
        quant = lambda t: " ".join(f"{float(v):.2f}" for v in torch.quantile(t.flatten().float(), torch.tensor([0.1, 0.5, 0.9, 0.99], device=t.device)))
        line = f"[krea2edit] guide step {i + 1}/{len(self.plan)} sigma {float(torch.as_tensor(sigma).flatten()[0]):.3f} early {self.plan[i][0]:.2f} late {self.plan[i][1]:.2f} | pull q10/50/90/99 {quant(w)}"
        maps = {"pull": w}
        if self.dist is not None:
            line += f" | dist {quant(self.dist)}"
            maps["dist"] = self.dist / 2
        if self.edit_acc is not None:
            line += f" | edit {quant(self.edit_acc)}"
            maps["edit"] = self.edit_acc
        print(line, flush=True)
        folder = os.path.join("output", "guide_lab_debug", self.tag)
        os.makedirs(folder, exist_ok=True)
        for name, t in maps.items():
            a = (t[0, 0].float().clamp(0, 1) * 255).byte().cpu().numpy()
            Image.fromarray(a).resize((a.shape[1] * 4, a.shape[0] * 4), Image.NEAREST).save(os.path.join(folder, f"{i + 1:02d}_{name}.png"))

    def step_of(self, sigma):
        s = float(torch.as_tensor(sigma).flatten()[0])
        return min(range(len(self.plan)), key=lambda i: abs(self.sigmas[i] - s))

    def needs_uncond(self, sigma):
        cfg = self.cfg
        if cfg.ref_guide <= 0 or cfg.edit_thr <= 0:
            return False
        s = float(torch.as_tensor(sigma).flatten()[0])
        return cfg.edit_end * self.sigmas[0] <= s <= cfg.edit_start * self.sigmas[0]

    def _remember(self, old, new):
        return new if old is None else self.cfg.smooth * old + (1 - self.cfg.smooth) * new

    def _distance_on(self):
        cfg = self.cfg
        return cfg.dist_calib > 0 or (cfg.dist_tau > 0 and cfg.ref_guide < 1)

    def _distance_freedom(self, x0, ref, coverage, mask):
        cfg = self.cfg
        if not self._distance_on():
            return 0.0
        if self.ref_std is None:
            self.ref_std = ref[coverage.expand_as(ref) > 0].float().std()
        d = token_rms(x0, ref) / self.ref_std
        self.dist = self._remember(self.dist, blur(d, round(cfg.dist_blur * TOKEN)))
        if cfg.dist_calib > 0:
            outside = ((mask > 0.5) & (coverage > 0)).expand_as(self.dist)
            background = self.dist[outside] if outside.any() else self.dist[(coverage > 0).expand_as(self.dist)]
            tau = cfg.dist_calib * torch.quantile(background.float(), 0.9).clamp(min=1e-3)
        else:
            tau = cfg.dist_tau * cfg.ref_guide / (1 - cfg.ref_guide)
        free = self.dist.pow(cfg.dist_pow) / (self.dist.pow(cfg.dist_pow) + tau ** cfg.dist_pow)
        if cfg.free_latch:
            self.free_max = free if self.free_max is None else torch.maximum(self.free_max, free)
            free = self.free_max
        if cfg.free_grow > 0:
            free = _grow(free, cfg.free_grow)
        return free

    def _update_edit_map(self, cond_x0, uncond_x0, mask):
        e = token_rms(cond_x0, uncond_x0)
        region = (mask < 0.5).expand_as(e)
        if self.cfg.edit_region_norm and region.any():
            scale = torch.stack([torch.quantile(b[r], self.cfg.edit_q) for b, r in zip(e, region)])
        else:
            scale = torch.quantile(e.flatten(1), self.cfg.edit_q, dim=1)
        e = (e / (scale.view(-1, 1, 1, 1) + 1e-6)).clamp(0, 1)
        self.edit_ema = self._remember(self.edit_ema, e)
        self.edit_acc = self.edit_ema if self.edit_acc is None else torch.maximum(self.edit_acc, self.edit_ema)

    def apply(self, sigma, x0, ref, mask, coverage, cond_x0=None, uncond_x0=None):
        """x0, ref: (B, C, H, W); mask, coverage: (1, 1, H, W) in 0..1. cond_x0 / uncond_x0 feed the edit map."""
        cfg = self.cfg
        i = self.step_of(sigma)
        early, late = self.plan[i]
        hold = self.hold[i]
        if early == 0 and late == 0 and hold == 0:
            return x0
        if self.needs_uncond(sigma) and cond_x0 is not None and uncond_x0 is not None:
            self._update_edit_map(cond_x0, uncond_x0, mask)
        free = self._distance_freedom(x0, ref, coverage, mask) if late > 0 else 0.0
        if self.edit_acc is not None:
            edit = _smoothstep((self.edit_acc - (cfg.edit_thr - cfg.edit_soft / 2)) / cfg.edit_soft)
            if cfg.edit_grow > 0:
                edit = _grow(edit, cfg.edit_grow)
            free = free * edit if cfg.gate_and and self._distance_on() else 1 - (1 - free) * (1 - edit)
        outside = mask
        if cfg.gate_in_mask:
            free = free * (1 - mask) ** cfg.gate_gamma
        mask = cfg.inside + (1 - cfg.inside) * mask
        if cfg.early_outside:
            w = blur(mask * late * (1 - free) + outside * early + (1 - outside) * hold, round(cfg.feather * min(x0.shape[-2:]))) * coverage
        else:
            w = blur(mask * (early + late * (1 - free)), round(cfg.feather * min(x0.shape[-2:]))) * coverage
        if cfg.debug:
            self._debug(i, sigma, w)
        return torch.lerp(x0, ref.to(x0.dtype), w.to(x0.dtype))


def add_guidance(m, cfg, reference, mask):
    """Install reference guidance on a patched model. `reference(H, W, mask)` yields (reference latent,
    coverage, mask weight) on the H x W target grid, or None when the reference is not aligned."""
    run = {}   # state of the current sampling call; dropped when it ends

    def sample(executor, guider, sigmas, *args):
        run["guide"] = Guide(cfg, sigmas)
        run["cfg1"] = math.isclose(guider.cfg, 1.0)
        run["placed"] = {}
        try:
            return executor(guider, sigmas, *args)
        finally:
            run.clear()

    def post_cfg(args):
        denoised = args["denoised"]
        guide = run.get("guide")
        if guide is None:
            return denoised
        x0 = _to_4d(denoised)
        size = tuple(x0.shape[-2:])
        if size not in run["placed"]:
            placed = reference(*size, mask)
            if placed is None:
                print("[krea2edit] WARNING: reference guidance needs ref_position_a 'aligned'; it is off.", flush=True)
            run["placed"][size] = placed if placed is None else tuple(t.to(x0.device) for t in placed)
        placed = run["placed"][size]
        if placed is None:
            return denoised
        ref, coverage, weight = placed
        cond = uncond = None
        if guide.needs_uncond(args["sigma"]) and args["uncond"] is not None:
            cond, uncond = _to_4d(args["cond_denoised"]), _to_4d(args["uncond_denoised"])
        out = guide.apply(args["sigma"], x0, ref[:1].expand(x0.shape[0], -1, -1, -1), weight, coverage, cond, uncond)
        return out.reshape(denoised.shape)

    def calc_cond(args):
        # the unconditional pass is forced for the whole run (disable_cfg1_optimization); drop it again
        # where the edit map does not need it and CFG is 1, which then returns the conditional prediction
        conds = args["conds"]
        guide = run.get("guide")
        if guide is not None and run["cfg1"] and not guide.needs_uncond(args["sigma"]):
            conds = [conds[0], None]
        return comfy.samplers.calc_cond_batch(args["model"], conds, args["input"], args["sigma"], args["model_options"])

    m.set_model_sampler_post_cfg_function(post_cfg, disable_cfg1_optimization=cfg.edit_thr > 0)
    if cfg.edit_thr > 0:
        m.set_model_sampler_calc_cond_batch_function(calc_cond)
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, "krea2_edit_advanced_guide", sample, m.model_options, is_model_options=True)


class Krea2EditAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2EditAdvanced",
            display_name="Krea2 Edit Advanced (source patch)",
            category="krea2edit",
            description="Adds the krea2_edit in-context source-preservation path (source latent as frame=1 tokens) to a Krea2 model.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("source_latent_a", tooltip="1st reference (the scene in two-image workflows, or the only image) -> RoPE frame=1. All *_a inputs control this reference"),
                io.Latent.Input("source_latent_b", optional=True, tooltip="2nd reference (the person) for multi-ref LoRAs -> RoPE frame=2, training-matched order: a first, b second. All *_b inputs control this reference"),
                io.Float.Input("ref_boost_a", default=1.0, min=0.0, max=1000.0, step=0.01, round=0.001, optional=True, tooltip="reference-fidelity dial for reference a: multiplies target->reference attention. 1.0 = off, >1 pulls harder toward the reference's appearance, <1 loosens. Optimal value is model-specific"),
                io.Float.Input("ref_boost_b", default=1.0, min=0.0, max=1000.0, step=0.01, round=0.001, optional=True, tooltip="same dial for reference b. No effect without a b reference. 1.0 = off"),
                io.Mask.Input("ref_boost_mask_a", optional=True, tooltip="optional region on reference a to boost, e.g. the face. Draw it on the same image you connect as reference a (any resolution); it follows the crop/resize the reference goes through. Gray values scale the boost proportionally (0 = no boost, 1 = full). Empty = whole reference"),
                io.Mask.Input("ref_boost_mask_b", optional=True, tooltip="same for reference b: drawn on the image you connect as reference b; gray values scale ref_boost_b proportionally. Empty = whole reference"),
                io.Combo.Input("ref_position_a", options=REF_POSITIONS, default=REF_POSITIONS[0], optional=True, tooltip="where reference a sits in position space. aligned (trained) = pinned onto the output layout, how the models were trained (default). beside = placed next to the canvas: identity still flows but the layout pin is gone, so the content can be restaged more freely. bag of features = experimental, content without layout"),
                io.Combo.Input("ref_position_b", options=REF_POSITIONS, default=REF_POSITIONS[0], optional=True, tooltip="the same choice for reference b; no effect without a b reference. For putting a person into a scene: a (scene) aligned + b (person) beside"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, source_latent_a, source_latent_b=None, ref_boost_a=1.0, ref_boost_b=1.0,
                ref_boost_mask_a=None, ref_boost_mask_b=None, ref_position_a="aligned (trained)",
                ref_position_b="aligned (trained)"):
        if not (ref_position_a.startswith("aligned") and ref_position_b.startswith("aligned")):
            print(f"[krea2edit] reference placement: a={ref_position_a!r}, "
                  f"b={ref_position_b!r}", flush=True)
        m = model.clone()
        # The target latent reaches the diffusion model already scaled (process_latent_in);
        # scale the source(s) the same way so all share one latent space.
        src = model.model.process_latent_in(source_latent_a["samples"])
        if source_latent_b is not None:
            src = [src, model.model.process_latent_in(source_latent_b["samples"])]

        def wrapper(executor, x, timesteps, context, *wargs, **kwargs):
            # ComfyUI signature drift (2026-07-19, commit c9602625 adds ref_latents):
            #   old: execute(x, t, ctx, attention_mask, transformer_options)
            #   new: execute(x, t, ctx, attention_mask, ref_latents, transformer_options)
            # Accept both: transformer_options is the trailing dict; any native
            # ref_latents are ignored (this patch supplies its own source path).
            transformer_options = kwargs.pop("transformer_options", None)
            if transformer_options is None:
                transformer_options = {}
                for a in reversed(wargs):
                    if isinstance(a, dict):
                        transformer_options = a
                        break
            dm = executor.class_obj  # the SingleStreamDiT instance
            return krea2_edit_forward(dm, x, timesteps, context, src, transformer_options,
                                      ref_boost_a=ref_boost_a, ref_boost_b=ref_boost_b,
                                      ref_boost_mask_a=ref_boost_mask_a, ref_boost_mask_b=ref_boost_mask_b,
                                      ref_position_a=ref_position_a, ref_position_b=ref_position_b)

        to = m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit_advanced", wrapper, to
        )
        return io.NodeOutput(m)


class Krea2EditAdvancedGroundedEncode(io.ComfyNode):
    """Image-grounded instruction encode — the SEMANTIC path of krea2_edit.

    Training always encodes the instruction TOGETHER with the source image through
    Qwen3-VL (user turn = <vision tokens: source> + instruction) and taps 12 layers.
    Stock CLIPTextEncode is text-only, so inference was running with the grounding
    half of the recipe missing (the VAE source tokens carry appearance; THIS carries
    scene semantics: "the man on the left", "the sign in the back").

    Requires a qwen3vl TE checkpoint WITH the vision tower (all local ones have it).
    grounding_px caps the longest side fed to the VLM — the 2026-07-02 LoRA trained
    with 384-768px jitter, so 640-768 is in-distribution; 0 = native res (the jitter
    training makes that tolerable too). For CFG, ground the NEGATIVE too: second node,
    empty prompt, same image (matches training's unconditional).
    """
    DEFAULT_SYSTEM = (
        "Describe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
    )

    KREA2_EDIT_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "{}<|im_end|>\n<|im_start|>assistant\n"
    )

    @classmethod
    def _template(cls, nimg, system_prompt=""):
        sp = system_prompt.strip() or cls.DEFAULT_SYSTEM
        vis = "<|vision_start|><|image_pad|><|vision_end|>" * nimg
        return ("<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
                + vis + "{}<|im_end|>\n<|im_start|>assistant\n")

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2EditAdvancedGroundedEncode",
            display_name="Krea2 Edit Advanced (grounded encode)",
            category="krea2edit",
            description="Encodes the edit instruction grounded on the source image (training-matched semantic path).",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Image.Input("image_a", optional=True, tooltip="1st reference (the scene, or the only image) — the image you encode into source_latent_a"),
                io.Image.Input("image_b", optional=True, tooltip="2nd reference (the person) for multi-ref LoRAs; vision blocks in training order: a, b"),
                io.Int.Input("grounding_px", default=768, min=0, max=4096, step=64, optional=True, tooltip="cap longest side fed to Qwen3-VL; 0 = native"),
                io.String.Input("system_prompt", multiline=True, default="", optional=True, tooltip="advanced (optional): override the grounding system prompt (empty = training default). Steers what the vision encoder attends to, e.g. facial identity detail."),
            ],
            outputs=[io.Conditioning.Output()],
        )

    KREA2_EDIT_TEMPLATE_2REF = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "{}<|im_end|>\n<|im_start|>assistant\n"
    )

    @classmethod
    def _prep(cls, image, grounding_px):
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            s = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    @classmethod
    def execute(cls, clip, prompt, image_a=None, image_b=None, grounding_px=768, system_prompt=""):
        if image_a is None:  # text-only fallback = old behavior
            tokens = clip.tokenize(prompt)
            return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))
        imgs = [cls._prep(image_a, grounding_px)]
        if image_b is not None:
            imgs.append(cls._prep(image_b, grounding_px))
        template = cls._template(len(imgs), system_prompt)
        tokens = clip.tokenize(prompt, images=imgs, llama_template=template)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class Krea2EditExtension(ComfyExtension):
    async def get_node_list(self):
        return [Krea2EditAdvanced, Krea2EditAdvancedGroundedEncode]


async def comfy_entrypoint():
    return Krea2EditExtension()


def _pack_version():
    # single source of truth = pyproject.toml, so this never drifts from the release
    try:
        import os, re
        p = os.path.join(os.path.dirname(__file__), "pyproject.toml")
        with open(p) as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


print(f"[krea2edit] nodes v{_pack_version()} loaded", flush=True)
