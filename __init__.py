"""ComfyUI-Krea2Edit — in-context edit forward for the Krea2 model.

ComfyUI's native Krea2 `_forward` is text-to-image only: it builds the sequence
`[text | target]`. The krea2_edit LoRA (trained in ai-toolkit) needs the *appearance
path*: the VAE-encoded SOURCE latent prepended as a block of clean tokens, distinguished
from the (noisy) target purely by the 3-axis RoPE frame index (source=1, target=0, h/w
aligned). This node adds that by wrapping the model's DIFFUSION_MODEL forward and rebuilding
the sequence as `[text | source(frame=1) | target(frame=0)]`, keeping only the target tokens
out — mirroring ai-toolkit's `predict_velocity_edit` exactly, using the model's own submodules.
On top of that the source hold pulls the sampler's denoised estimate back to the source latent (optionally
limited by an edit mask), so edits do not shift or darken the rest of the picture.

Wiring:  LoadImage -> VAEEncode(source) --\
                                            Krea2EditAdvanced(model, source_latent_a) -> KSampler
         UNETLoader -> LoraLoaderModelOnly -/
KSampler.latent_image <- EmptySD3LatentImage (noise). Text: NATIVE krea2 CLIP + CLIPTextEncode.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.patcher_extension
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


def _patchify(x, patch):
    """(b, c, h * patch, w * patch) -> (b, h * w, c * patch * patch)."""
    b, c, H, W = x.shape
    return x.reshape(b, c, H // patch, patch, W // patch, patch).permute(0, 2, 4, 1, 3, 5).reshape(b, (H // patch) * (W // patch), c * patch * patch)


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
    source. Returns the reference and the mask (drawn on the reference) averaged onto that grid, cropped to H x W."""
    Hp, Wp = -(-H // TOKEN) * TOKEN, -(-W // TOKEN) * TOKEN
    ref = comfy.ldm.common_dit.pad_to_patch_size(_to_4d(ref), (TOKEN, TOKEN), padding_mode="replicate")
    if ref.shape[-2:] != (Hp, Wp):
        mask = _crop_mask(mask, *ref.shape[-2:], _fit_box(*ref.shape[-2:], Hp, Wp))
        ref = _fit_src(ref, Hp, Wp)
    weight = F.interpolate((mask if mask.ndim == 3 else mask[None])[:1][None].float(), size=(Hp, Wp), mode="area")
    return ref[..., :H, :W], weight[..., :H, :W]


DISC = 24          # largest disc radius (pixels) filtered exactly; a larger one is filtered on a coarser grid


def _disc_max(m, r):
    """Maximum of m (1, 1, H, W) over a disc of radius r pixels: one 1-D max-pool per row of the disc. The border repeats its edge pixels."""
    if r <= 0:
        return m
    h, w = m.shape[-2:]
    f = max(1, math.ceil(r / DISC))
    if f > 1:
        m = F.max_pool2d(m, f, ceil_mode=True)                  # max keeps thin strokes; an average would dilute them
    hl, wl = m.shape[-2:]
    r = max(1, round(r / f))
    p = F.pad(m, (r, r, r, r), mode="replicate")
    out = None
    for dy in range(-r, r + 1):
        half = int((r * r - dy * dy) ** 0.5)
        row = F.max_pool2d(p[..., r + dy:r + dy + hl, :], (1, 2 * half + 1), 1, (0, half))[..., r:r + wl]
        out = row if out is None else torch.maximum(out, row)
    return out if f == 1 else F.interpolate(out, size=(h, w), mode="bilinear")


def _gaussian(m, sigma):
    """Gaussian blur of m (1, 1, H, W), sigma in pixels, edge pixels repeated; a large sigma is blurred on a coarser grid."""
    if sigma < 0.5:
        return m
    h, w = m.shape[-2:]
    f = max(1, int(sigma // 8))
    if f > 1:
        m = F.interpolate(m, size=(-(-h // f), -(-w // f)), mode="area")
    sigma = sigma / f
    r = math.ceil(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, dtype=m.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    m = F.conv2d(F.pad(m, (r, r, r, r), mode="replicate"), k.view(1, 1, 1, -1))
    m = F.conv2d(m, k.view(1, 1, -1, 1))
    return m if f == 1 else F.interpolate(m, size=(h, w), mode="bilinear")


def _edit_mask(mask, size, grow_percent, feather_percent):
    """The edit area on the pixel grid `size` = (h, w): the drawn MASK (white everywhere without one) grown (positive) or shrunk
    (negative) by a disc, then feathered; both in percent of the shorter side. Returns (1, h, w) in 0..1."""
    h, w = size
    if mask is None:
        m = torch.ones(1, 1, h, w)
    else:
        m = (mask if mask.ndim == 3 else mask[None])[:1][None].float()
        m = F.interpolate(m, size=(h, w), mode="area" if m.shape[-2] >= h else "bilinear")
    r = grow_percent * min(h, w) / 100
    m = _disc_max(m, round(r)) if r > 0 else 1 - _disc_max(1 - m, round(-r))
    return _gaussian(m, feather_percent * min(h, w) / 100).clamp(0, 1)[0]


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
            sh, sw = src.shape[-2:]
            if masks[i] is not None:
                masks[i] = _crop_mask(masks[i], sh, sw, _fit_box(sh, sw, H, W))
            src = _fit_src(src, H, W).to(x.dtype)
        srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch), padding_mode="replicate"))
    src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

    context = m._unpack_context(context)                       # (B, seq, 12, 2560)

    tgt_img = m.first(_patchify(x, patch))
    src_imgs = [m.first(_patchify(s_, patch)) for s_ in srcs]

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
    out = out.reshape(bs, h_, w_, m.channels, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(bs, m.channels, h_ * patch, w_ * patch)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


# ---------------------------------------------------------------------------------------------------------
# Source hold: pull the denoised estimate toward the source latent.
#
# At every step the sampler's x0 estimate is blended toward the source, x0' = x0 + w * (src - x0). This is
# the same update as blending the velocity toward the source's straight-line velocity (RF-Inversion's
# controller guidance). Output preservation is decided by the late steps, layout by the early ones, so the
# schedule has two phases split at `crossover * sigma_start`. Where the mask is 0 (the edit area) the late
# pull is gated by distance: tokens whose estimate is far from the source are left alone, so an edit that
# the model clearly wants to make is not pulled back, while the rest of the edit area is held.
# ---------------------------------------------------------------------------------------------------------
TOKEN = 2  # latent cells per token side (patch size)


@dataclass
class GuideConfig:
    ref_guide: float = 0.9     # pull strength 0..1; 1 = copy the source where the weight is full
    crossover: float = 0.5     # early phase = sigma above crossover * sigma_start
    late_gain: float = 10.0    # late-phase strength = min(1, late_gain * ref_guide)
    early_gain: float = 0.8    # early-phase strength outside the edit area = early_gain * ref_guide
    early_hold: float = 0.0    # hold knob: the early pull inside the edit area peaks at min(X, 1) and falls with sigma with exponent 2 / X; 0 = none
    dist_tau: float = 0.02     # distance-gate tolerance in source-std units; 0 = off
    dist_pow: float = 8.0      # steepness n of the gate d^n / (d^n + tol^n)
    smooth: float = 0.5        # across-step memory of the distance map (0 = none)


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


class Guide:
    """State of one sampling run. Build a new one per run."""

    def __init__(self, cfg, sigmas):
        self.cfg = cfg
        self.sigmas = [float(s) for s in sigmas]
        self.plan = step_weights(self.sigmas, cfg.ref_guide, cfg.crossover, cfg.late_gain, cfg.early_gain)
        self.hold = [e for e, _ in step_weights(self.sigmas, cfg.ref_guide, cfg.crossover, cfg.late_gain, min(cfg.early_hold, 1.0), 2.0 / cfg.early_hold)] if cfg.early_hold > 0 else [0.0] * len(self.plan)
        self.ref_std = None
        self.dist = None

    def step_of(self, sigma):
        s = float(torch.as_tensor(sigma).flatten()[0])
        return min(range(len(self.plan)), key=lambda i: abs(self.sigmas[i] - s))

    def _distance_freedom(self, x0, ref):
        cfg = self.cfg
        if cfg.dist_tau <= 0 or cfg.ref_guide >= 1:
            return 0.0
        if self.ref_std is None:
            self.ref_std = ref.float().std()
        d = token_rms(x0, ref) / self.ref_std
        self.dist = d if self.dist is None else cfg.smooth * self.dist + (1 - cfg.smooth) * d
        tau = cfg.dist_tau * cfg.ref_guide / (1 - cfg.ref_guide)
        return self.dist.pow(cfg.dist_pow) / (self.dist.pow(cfg.dist_pow) + tau ** cfg.dist_pow)

    def apply(self, sigma, x0, ref, mask):
        """x0, ref: (B, C, H, W); mask: (1, 1, H, W) in 0..1, 1 = keep as the source, 0 = edit area."""
        i = self.step_of(sigma)
        early, late = self.plan[i]
        hold = self.hold[i]
        if early == 0 and late == 0 and hold == 0:
            return x0
        free = self._distance_freedom(x0, ref) * (1 - mask) if late > 0 else 0.0
        w = late * (1 - free) + mask * early + (1 - mask) * hold
        return torch.lerp(x0, ref.to(x0.dtype), w.to(x0.dtype))


def add_guidance(m, cfg, source, mask):
    """Install the source hold on a patched model. `source` is the source latent (scaled like the target), `mask` the
    hold weight on the pixel grid (1 = keep as the source)."""
    run = {}   # state of the current sampling call; dropped when it ends

    def sample(executor, guider, sigmas, *args):
        run["guide"] = Guide(cfg, sigmas)
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
            run["placed"][size] = tuple(t.to(x0.device) for t in _place_ref(source, mask, *size))
        ref, weight = run["placed"][size]
        out = guide.apply(args["sigma"], x0, ref[:1].expand(x0.shape[0], -1, -1, -1), weight)
        return out.reshape(denoised.shape)

    m.set_model_sampler_post_cfg_function(post_cfg)
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, "krea2_edit_advanced_guide", sample, m.model_options, is_model_options=True)


class Krea2EditAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2EditAdvanced",
            display_name="Krea2 Edit Advanced (source patch)",
            category="krea2edit",
            description="Lets a Krea 2 edit model see the source picture(s) while it draws and holds the picture to the source where the edit should not change it.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("source_latent_a", tooltip="The source picture, encoded with VAE Encode. The model sees it while it draws, and source_hold pulls the picture back to it. It must be the same size as the picture being drawn (the latent you give the sampler): otherwise it is cropped to the shape of the output and resized in latent space, which loses parts of the image and softens details, and with source_hold the softened version is pulled into the result. Scale the image to the output size before VAE Encode, or use the encoded source as the sampler's latent."),
                io.Latent.Input("source_latent_b", optional=True, tooltip="Optional second source picture, for example a person to put into the scene. The model sees it next to source A; the source hold does not use it. Needs a LoRA trained for two sources."),
                io.Float.Input("ref_boost_a", default=1.0, min=0.0, max=1000.0, step=0.01, round=0.001, optional=True, tooltip="How closely the drawing follows the look of source A. 1 = normal. Higher = sticks closer to the source's details (too high copies the source and weakens the edit), lower = looser. Applies to all of source A, or only to the part marked in ref_boost_mask_a."),
                io.Float.Input("ref_boost_b", default=1.0, min=0.0, max=1000.0, step=0.01, round=0.001, optional=True, tooltip="The same for source B. No effect without source B. 1 = normal."),
                io.Mask.Input("ref_boost_mask_a", optional=True, tooltip="Limits ref_boost_a to a part of source A, for example the face: white = full boost, black = none, gray = partly. Draw it on the source image, any resolution. Empty = the whole source."),
                io.Mask.Input("ref_boost_mask_b", optional=True, tooltip="The same for source B and ref_boost_b."),
                io.DynamicCombo.Input("ref_position_a", tooltip="How source A is shown to the model. aligned (trained) = laid exactly over the picture being drawn, the way the models were trained; the source hold and edit mask need it. beside = placed next to the picture: the look carries over but the layout is free, so things can be rearranged. bag of features = experimental, the look without any layout.", options=[
                    io.DynamicCombo.Option(REF_POSITIONS[0], [
                        io.Boolean.Input("source_hold", default=True, tooltip="Pulls the picture back towards source A while it is drawn, wherever the edit should not change it. Off = no pull, the model alone decides, which can shift the picture by a pixel or two and darken it. If you plan several edits in a row, pass the output latent of one step as the source of the next. Works on the whole picture."),
                        io.Float.Input("hold_strength", default=0.7, min=0.0, max=10.0, step=0.05, tooltip="How firmly the edit area stays close to source A early in the drawing, while the layout forms. The edit area is the white part of edit_mask, or the whole picture when there is no mask; outside the mask the source is always held firmly. Higher = steadier picture (less shift, darkening or texture change) but smaller, more source-bound edits; lower = more freedom for the edit. 0.7 is a balance. At 0.5 the picture still drifts noticeably; at 4 added objects come out about half as big. 0 = no extra hold in the edit area."),
                        io.Mask.Input("edit_mask", optional=True, tooltip="Where the picture may change: white = edit area, black = keep as the source, gray = partly. Draw it on the source image, any resolution. Without a mask the whole picture is the edit area. Krea 2 does not use the mask to choose where objects appear, so say where in the prompt. Best workflow: run once without a mask to see where Krea 2 puts the object, then draw the mask there and run again."),
                        io.Float.Input("grow_percent", default=0.0, min=-100.0, max=100.0, step=0.5, tooltip="Makes the white area of edit_mask (where the edit may change the picture) bigger (positive) or smaller (negative) all around its edge, in percent of the shorter side of the picture. 0 = the mask as drawn."),
                        io.Float.Input("feather_percent", default=0.0, min=0.0, max=100.0, step=0.5, tooltip="Softens the edge of the white area of edit_mask, in percent of the shorter side of the picture, after growing or shrinking. 0 = hard edge."),
                    ]),
                    io.DynamicCombo.Option(REF_POSITIONS[1], []),
                    io.DynamicCombo.Option(REF_POSITIONS[2], []),
                ]),
                io.Combo.Input("ref_position_b", options=REF_POSITIONS, default=REF_POSITIONS[0], optional=True, tooltip="The same choice for source B; no effect without source B. To put a person into a scene: A (the scene) aligned, B (the person) beside."),
            ],
            outputs=[io.Model.Output(), io.Mask.Output(tooltip="edit_mask as the node uses it, after growing and feathering (white = where the edit may change the picture; all white without edit_mask).")],
        )

    @classmethod
    def execute(cls, model, source_latent_a, ref_position_a, source_latent_b=None, ref_boost_a=1.0, ref_boost_b=1.0,
                ref_boost_mask_a=None, ref_boost_mask_b=None, ref_position_b="aligned (trained)"):
        position_a = ref_position_a["ref_position_a"]
        if not (position_a.startswith("aligned") and ref_position_b.startswith("aligned")):
            print(f"[krea2edit] reference placement: a={position_a!r}, "
                  f"b={ref_position_b!r}", flush=True)
        m = model.clone()
        # The target latent reaches the diffusion model already scaled (process_latent_in);
        # scale the source(s) the same way so all share one latent space.
        src = model.model.process_latent_in(source_latent_a["samples"])
        if source_latent_b is not None:
            src = [src, model.model.process_latent_in(source_latent_b["samples"])]

        source_size = tuple(_to_4d(source_latent_a["samples"]).shape[-2:])
        warned = []

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
            if not warned and tuple(_to_4d(x).shape[-2:]) != source_size:
                warned.append(True)
                print(f"[krea2edit] WARNING: source_latent_a is {source_size[1] * 8}x{source_size[0] * 8}px but the picture is drawn at "
                      f"{x.shape[-1] * 8}x{x.shape[-2] * 8}px: the source is cropped and resized in latent space, which softens it. "
                      f"Scale the image to the output size before VAE Encode.", flush=True)
            return krea2_edit_forward(dm, x, timesteps, context, src, transformer_options,
                                      ref_boost_a=ref_boost_a, ref_boost_b=ref_boost_b,
                                      ref_boost_mask_a=ref_boost_mask_a, ref_boost_mask_b=ref_boost_mask_b,
                                      ref_position_a=position_a, ref_position_b=ref_position_b)

        to = m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit_advanced", wrapper, to
        )
        h, w = _to_4d(source_latent_a["samples"]).shape[-2:]
        edit = _edit_mask(ref_position_a.get("edit_mask"), (h * 8, w * 8), ref_position_a.get("grow_percent", 0.0), ref_position_a.get("feather_percent", 0.0))
        if ref_position_a.get("source_hold"):
            add_guidance(m, GuideConfig(early_hold=ref_position_a["hold_strength"]), src[0] if isinstance(src, list) else src, 1 - edit)
        return io.NodeOutput(m, edit)


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
            description="Encodes the instruction together with the source picture, the way the edit models were trained, so the text can refer to what is in the picture.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default="", tooltip="What to change. Krea 2 Edit does not use the mask to place things, so say where the new object goes; long prompts are fine. Leave empty on the negative encode."),
                io.Image.Input("image_a", optional=True, tooltip="The source picture, the one you encode into source_latent_a. Connect the same picture to the negative encode as well."),
                io.Image.Input("image_b", optional=True, tooltip="Optional second source picture, the same one you encode into source_latent_b."),
                io.Int.Input("grounding_px", default=768, min=0, max=4096, step=64, optional=True, tooltip="Longest side, in pixels, of the picture the text encoder looks at; larger pictures are scaled down for it only, not for the output. 0 = full size. 640-768 matches the training."),
                io.String.Input("system_prompt", multiline=True, default="", optional=True, tooltip="Optional. Replaces the instruction that tells the text encoder what to look at (empty = the one the models were trained with), for example to stress facial detail."),
            ],
            outputs=[io.Conditioning.Output()],
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
