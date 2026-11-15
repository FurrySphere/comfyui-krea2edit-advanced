"""ComfyUI-Krea2Edit — in-context edit forward for the Krea2 model.

ComfyUI's native Krea2 `_forward` is text-to-image only: it builds the sequence
`[text | target]`. The krea2_edit LoRA (trained in ai-toolkit) needs the *appearance
path*: the VAE-encoded SOURCE latent prepended as a block of clean tokens, distinguished
from the (noisy) target purely by the 3-axis RoPE frame index (source=1, target=0, h/w
aligned). This node adds that by wrapping the model's DIFFUSION_MODEL forward and rebuilding
the sequence as `[text | source(frame=1) | target(frame=0)]`, keeping only the target tokens
out — mirroring ai-toolkit's `predict_velocity_edit` exactly, using the model's own submodules.

Wiring:  LoadImage -> VAEEncode(source) --\
                                            Krea2EditModelPatch(model, source_latent_a) -> KSampler
         UNETLoader -> LoraLoaderModelOnly -/
KSampler.latent_image <- EmptySD3LatentImage (noise). Text: NATIVE krea2 CLIP + CLIPTextEncode.
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.patcher_extension
import comfy.utils
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    """Stride-1 integer positions at a centered integer offset. For `fit` refs the
    pixels are already resampled to target grid density, so the position grid is
    stride-1 BY CONSTRUCTION — scaling it again only manufactures skip/collision
    artifacts. Requires gh<=th, gw<=tw (guaranteed by the floor+cap in fit)."""
    # fractional center (2026-07-28): integer floor placed odd-gap refs 8px off
    # their true center — RoPE is continuous, half-token positions are exact.
    off_h, off_w = max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


RENAMED_INPUTS = {"source_latent": "source_latent_a", "source_image": "source_image_a", "image": "image_a",
                  "ref_boost": "ref_boost_a (one image) / ref_boost_b (two images)",
                  "ref_boost_mask": "ref_boost_mask_a (one image) / ref_boost_mask_b (two images)",
                  "ref_position": "ref_position_a (one image) / ref_position_b (two images)"}


def _reject_renamed(inputs):
    """A workflow saved before the _a/_b rename still sends the old names; ignoring them would silently drop the references."""
    old = [f"{k} -> {RENAMED_INPUTS[k]}" for k in inputs if k in RENAMED_INPUTS]
    if old:
        raise ValueError("Krea2Edit inputs were renamed, reconnect them (see CHANGELOG.md): " + "; ".join(old))


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


def _fit_box(ih, iw, px_h, px_w, fit_mode):
    """Window of an ih x iw source that becomes the reference for a px_h x px_w target, and the size
    it is resized to: (y0, x0, crop_h, crop_w, out_h, out_w). Reference images, source latents and
    their boost masks all follow this one geometry."""
    if fit_mode == "fit":
        # "bilinear" answer to scale mismatch: resample CONTENT (pixel space, bicubic)
        # to the target's grid density instead of moving positions. AR-preserving
        # fit-inside, no crop, no grey canvas — the forward places it at an integer
        # centered offset (scaled-pos with s=1 -> stride 1, no rounding artifacts).
        sc = min(px_h / ih, px_w / iw)
        # NEAR-MATCHED AR: fill the target grid EXACTLY via a minimal center-crop.
        # Fit-inside margins of 1-2 tokens are not harmless: target edge columns
        # with no ref correspondence get filled by repeating adjacent ref content
        # (2026-07-14 edge-duplication bug: ref (74,54) vs target (74,56)).
        # This also restores the design promise fit == crop at matched AR.
        CROP_TOL = 0.08
        if not (ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL)):
            # genuine AR mismatch: MUST match the trainer's _fit_prep EXACTLY
            # (krea2_edit.py) — /16 floor snap capped at the target's /16 floor.
            # The model is trained on this geometry; a /8-round node grid would
            # produce a different ref latent size -> different centered offset ->
            # a visible margin-boundary seam even from a well-trained model
            # (train/infer geometry must be byte-identical). 2026-07-15 alignment.
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            # CROP-TO-GRID (2026-07-28 seam-doubling RCA): resizing ih*sc -> floor16
            # SQUASHES content by up to 15px; the misregistration peaks exactly at
            # the ref band edges — the outpaint seam — and renders as a doubled
            # band (proven causal: 754px vs 753px input A/B, one pixel flips
            # clean<->worst). Center-crop the source so the fitted axis lands on
            # the /16 grid at scale sc EXACTLY: zero squash, stride-1 stays true.
            ch, cw = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            return (ih - ch) // 2, (iw - cw) // 2, ch, cw, nh, nw
    # crop (default / "v1 legacy", and fit at near-matched AR): center-crop to the target AR.
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    return (ih - ch) // 2, (iw - cw) // 2, ch, cw, px_h, px_w


def _crop_mask(mask, ih, iw, box):
    """Crop a MASK drawn on an ih x iw source to `box` (y0, x0, h, w) of that source. The mask may
    have a different resolution than the source it was drawn on."""
    y0, x0, ch, cw = box
    mh, mw = mask.shape[-2:]
    ya, xa = round(y0 * mh / ih), round(x0 * mw / iw)
    yb, xb = max(ya + 1, round((y0 + ch) * mh / ih)), max(xa + 1, round((x0 + cw) * mw / iw))
    return mask[..., ya:yb, xa:xb]


def _crop_mask_to_image(mask, image, H, W, fit_mode):
    """The mask of a reference image, cropped to the part of the image _fit_encode_image keeps."""
    if mask is None or image is None:
        return mask
    ih, iw = image.shape[1:3]
    return _crop_mask(mask, ih, iw, _fit_box(ih, iw, H * 8, W * 8, fit_mode)[:4])


def _fit_src(src, H, W):
    """Fit a source latent to the target grid the way TRAINING did: center-crop to
    the target aspect ratio, then resize. A plain interpolate (the pre-fix behavior)
    STRETCHES mixed-AR sources — users saw stretched people whenever their input AR
    differed from the output resolution."""
    sh, sw = src.shape[-2:]
    if (sh, sw) == (H, W):
        return src
    y0, x0, ch, cw, _, _ = _fit_box(sh, sw, H, W, "crop")
    src = src[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(src.float(), size=(H, W), mode="bilinear")


def _fit_encode_image(image, vae, H, W, cache, key, fit_mode="crop"):
    """Pixel-space source prep (blur-proof path): crop and resize the IMAGE to the
    target grid per _fit_box, VAE-encode. Latent-space resizing (the old fallback)
    softens VAE latents — this path never resizes latents at all. Cached per target
    resolution (encode once, not per step)."""
    key = key + (fit_mode,)
    if key in cache:
        return cache[key]
    print(f"[krea2edit] _fit_encode_image: mode={fit_mode} in={tuple(image.shape)} target_latent={H}x{W}", flush=True)
    img = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    y0, x0, ch, cw, nh, nw = _fit_box(*img.shape[-2:], H * 8, W * 8, fit_mode)
    img = F.interpolate(img[..., y0:y0 + ch, x0:x0 + cw].float(), size=(nh, nw), mode="bicubic", antialias=True)
    lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
    cache[key] = lat
    return lat


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
                       ref_native=False, pos_mode="anchor", ref_position_a="aligned (trained)",
                       ref_position_b="aligned (trained)"):
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
        if not ref_native and src.shape[-2:] != (H, W):
            print(f"[krea2edit] LATENT-PATH fit_src (crop): src={tuple(src.shape[-2:])} -> {H}x{W}", flush=True)
            sh, sw = src.shape[-2:]
            if masks[i] is not None:
                masks[i] = _crop_mask(masks[i], sh, sw, _fit_box(sh, sw, H, W, "crop")[:4])
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
    if pos_mode == "stride1" and ref_native:
        print(f"[krea2edit] STRIDE1-POS fit: ref grids {src_grids} centered in ({h_},{w_})", flush=True)
        ref_ids = [_imgids_offset(bs, i + 1, gh, gw, h_, w_, device)
                   for i, (gh, gw) in enumerate(src_grids)]
    else:
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


class Krea2EditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
        }, "optional": {
            "source_latent_a": ("LATENT", {"tooltip": "1st reference (the scene in two-image workflows, or the only image) -> RoPE frame=1. All *_a inputs control this reference. Not needed (and ignored) when vae + source_image_a are connected"}),
            "source_latent_b": ("LATENT", {"tooltip": "2nd reference (the person) for multi-ref LoRAs -> RoPE frame=2, training-matched order: a first, b second. All *_b inputs control this reference. Ignored when vae + source_image_a are connected"}),
            "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                       "tooltip": "reference-fidelity dial for reference a: multiplies target->reference attention. 1.0 = off, >1 pulls harder toward the reference's appearance, <1 loosens. Optimal value is model-specific"}),
            "ref_boost_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "round": 0.001,
                                       "tooltip": "same dial for reference b. No effect without a b reference. 1.0 = off"}),
            "fit_mode": (["fit", "crop (legacy)"], {"default": "fit",
                          "tooltip": "how an image source fits a mismatched output aspect ratio (needs vae + source_image_a connected): fit = resample the source to the target grid at a centered offset — matches how this model was trained (default, use this); crop (legacy) = center-crop to the target AR then resize (v1/v1.1 geometry, only for older weights)"}),
            "ref_boost_mask_a": ("MASK", {"tooltip": "optional region on reference a to boost, e.g. the face. Draw it on the same image you connect as reference a (any resolution); it follows the crop/resize the reference goes through. Gray values scale the boost proportionally (0 = no boost, 1 = full). Empty = whole reference"}),
            "ref_boost_mask_b": ("MASK", {"tooltip": "same for reference b: drawn on the image you connect as reference b; gray values scale ref_boost_b proportionally. Empty = whole reference"}),
            "vae": ("VAE", {"tooltip": "RECOMMENDED with source_image_a: enables the blur-proof pixel-space path (crop+resize in pixels, encode internally) — immune to input/output resolution mismatches"}),
            "source_image_a": ("IMAGE", {"tooltip": "reference a as IMAGE (with vae connected): overrides source_latent_a (and source_latent_b) with exact pixel-space fitting — fixes blurry results from mismatched resolutions"}),
            "source_image_b": ("IMAGE", {"tooltip": "reference b as IMAGE (with vae and source_image_a)"}),
            # declared last on purpose: appending keeps every existing socket index put,
            # so workflows saved before this input still reload with their links intact.
            "target_latent": ("LATENT", {"tooltip": "RECOMMENDED with vae + source_image_a: wire the SAME latent you feed KSampler.latent_image. Lets the node VAE-encode the source here, before sampling starts, instead of on the first step — otherwise the VAE is pulled onto the GPU mid-sampling and can evict part of the diffusion model, slowing every remaining step on VRAM-tight setups"}),
            # appended last, same reason as target_latent
            "ref_position_a": (REF_POSITIONS, {"default": REF_POSITIONS[0],
                               "tooltip": "where reference a sits in position space. aligned (trained) = pinned onto the output layout, how the models were trained (default). beside = placed next to the canvas: identity still flows but the layout pin is gone, so the content can be restaged more freely. bag of features = experimental, content without layout"}),
            "ref_position_b": (REF_POSITIONS, {"default": REF_POSITIONS[0],
                               "tooltip": "the same choice for reference b; no effect without a b reference. For putting a person into a scene: a (scene) aligned + b (person) beside"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Adds the krea2_edit in-context source-preservation path (source latent as frame=1 tokens) to a Krea2 model."

    def patch(self, model, source_latent_a=None, source_latent_b=None, ref_boost_a=1.0, ref_boost_b=1.0,
              ref_boost_mask_a=None, ref_boost_mask_b=None, vae=None, source_image_a=None,
              source_image_b=None, fit_mode="fit", target_latent=None,
              ref_position_a="aligned (trained)", ref_position_b="aligned (trained)", **_future):
        _reject_renamed(_future)
        if _future:
            print(f"[krea2edit] WARNING: workflow provides inputs this node version does not "
                  f"know ({', '.join(sorted(_future))}). The workflow is newer than the "
                  f"installed node pack. Update comfyui-krea2edit (Manager -> Update, or git "
                  f"pull) and restart ComfyUI. Continuing without them.", flush=True)
        if not (ref_position_a.startswith("aligned") and ref_position_b.startswith("aligned")):
            print(f"[krea2edit] reference placement: a={ref_position_a!r}, "
                  f"b={ref_position_b!r}", flush=True)
        m = model.clone()
        pixel_path = vae is not None and source_image_a is not None
        src_samples = None
        if not pixel_path:
            if source_latent_a is None:
                raise ValueError("Krea2EditModelPatch: connect source_latent_a, or vae + source_image_a")
            # The target latent reaches the diffusion model already scaled (process_latent_in);
            # scale the source(s) the same way so all share one latent space.
            src_samples = model.model.process_latent_in(source_latent_a["samples"])
            if source_latent_b is not None:
                src_samples = [src_samples, model.model.process_latent_in(source_latent_b["samples"])]

        px_cache = {}   # pixel-path encoded sources, keyed per target resolution
        mm = model.model  # for process_latent_in on the pixel path
        state = {"announced": False}

        if fit_mode == "fit" and not pixel_path:
            print(f"[krea2edit] WARNING: fit_mode='fit' has NO EFFECT — it needs both "
                  f"'vae' and 'source_image_a' connected (the pixel path). Falling back to the "
                  f"latent crop path.", flush=True)

        # Pre-encode OUTSIDE the sampling window. vae.encode -> load_models_gpu ->
        # free_memory(keep_loaded=[]), which partially unloads whatever is resident —
        # including the diffusion model, if the first call lands inside the sampler.
        # Nothing re-expands it (sampler_helpers loads once, before the loop), so the
        # rest of the run streams weights from CPU every step. Running the encode here,
        # at node-execution time, restores the ordinary VAEEncode -> KSampler order
        # where the sampler evicts the VAE instead of the reverse.
        primed = None
        if pixel_path:
            if target_latent is not None:
                Hh, Ww = target_latent["samples"].shape[-2], target_latent["samples"].shape[-1]
                print(f"[krea2edit] pre-encoding sources at target {Hh * 8}x{Ww * 8}px "
                      f"(before sampling, fit_mode={fit_mode})", flush=True)
                _fit_encode_image(source_image_a, vae, Hh, Ww, px_cache, ("a", Hh, Ww), fit_mode)
                if source_image_b is not None:
                    _fit_encode_image(source_image_b, vae, Hh, Ww, px_cache, ("b", Hh, Ww), fit_mode)
                primed = (Hh, Ww)
            else:
                print("[krea2edit] NOTE: connect 'target_latent' (the same latent that feeds "
                      "KSampler.latent_image) to pre-encode the source here instead of on the "
                      "first sampling step. Without it the VAE is loaded mid-sampling and can "
                      "evict part of the diffusion model, slowing every remaining step.",
                      flush=True)

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
            src = src_samples
            if pixel_path:
                xx = _to_4d(x)
                Hh, Ww = xx.shape[-2], xx.shape[-1]
                # not `if not px_cache` — the cache is already primed when target_latent
                # is wired, so an explicit one-shot flag is needed to announce once.
                if not state["announced"]:
                    state["announced"] = True
                    print(f"[krea2edit] pixel path ACTIVE (fit_mode={fit_mode})", flush=True)
                    if primed is not None and primed != (Hh, Ww):
                        print(f"[krea2edit] WARNING: 'target_latent' is {primed[0] * 8}x{primed[1] * 8}px but "
                              f"sampling is at {Hh * 8}x{Ww * 8}px — the pre-encode is unused and the VAE "
                              f"will run mid-sampling. Wire the SAME latent that feeds KSampler.",
                              flush=True)
                lat = mm.process_latent_in(_fit_encode_image(source_image_a, vae, Hh, Ww, px_cache, ("a", Hh, Ww), fit_mode))
                if source_image_b is not None:
                    lat = [lat, mm.process_latent_in(_fit_encode_image(source_image_b, vae, Hh, Ww, px_cache, ("b", Hh, Ww), fit_mode))]
                src = lat
            mask_a, mask_b = ref_boost_mask_a, ref_boost_mask_b
            if pixel_path:
                mask_a = _crop_mask_to_image(mask_a, source_image_a, Hh, Ww, fit_mode)
                mask_b = _crop_mask_to_image(mask_b, source_image_b, Hh, Ww, fit_mode)
            v = krea2_edit_forward(dm, x, timesteps, context, src, transformer_options,
                                   ref_boost_a=ref_boost_a, ref_boost_b=ref_boost_b,
                                   ref_boost_mask_a=mask_a, ref_boost_mask_b=mask_b,
                                   ref_native=(fit_mode == "fit" and pixel_path),
                                   pos_mode=("stride1" if fit_mode == "fit" else "anchor"),
                                   ref_position_a=ref_position_a, ref_position_b=ref_position_b)
            return v

        to = m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit", wrapper, to
        )
        return (m,)


class Krea2EditGroundedEncode:
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image_a": ("IMAGE", {"tooltip": "1st reference (the scene, or the only image) — the same image as source_image_a / source_latent_a"}),
                "image_b": ("IMAGE", {"tooltip": "2nd reference (the person) for multi-ref LoRAs; vision blocks in training order: a, b"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64,
                                          "tooltip": "cap longest side fed to Qwen3-VL; 0 = native"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "advanced (optional): override the grounding system prompt (empty = training default). Steers what the vision encoder attends to, e.g. facial identity detail."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Encodes the edit instruction grounded on the source image (training-matched semantic path)."

    KREA2_EDIT_TEMPLATE_2REF = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "{}<|im_end|>\n<|im_start|>assistant\n"
    )

    def _prep(self, image, grounding_px):
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            s = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(self, clip, prompt, image_a=None, image_b=None, grounding_px=768, system_prompt="", **_future):
        _reject_renamed(_future)
        if _future:
            print(f"[krea2edit] WARNING: workflow provides inputs this node version does not "
                  f"know ({', '.join(sorted(_future))}). Update comfyui-krea2edit and restart "
                  f"ComfyUI. Continuing without them.", flush=True)
        if image_a is None:  # text-only fallback = old behavior
            tokens = clip.tokenize(prompt)
            return (clip.encode_from_tokens_scheduled(tokens),)
        imgs = [self._prep(image_a, grounding_px)]
        if image_b is not None:
            imgs.append(self._prep(image_b, grounding_px))
        template = self._template(len(imgs), system_prompt)
        tokens = clip.tokenize(prompt, images=imgs, llama_template=template)
        return (clip.encode_from_tokens_scheduled(tokens),)


NODE_CLASS_MAPPINGS = {
    "Krea2EditModelPatch": Krea2EditModelPatch,
    "Krea2EditGroundedEncode": Krea2EditGroundedEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditModelPatch": "Krea2 Edit (source patch)",
    "Krea2EditGroundedEncode": "Krea2 Edit (grounded encode)",
}


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
