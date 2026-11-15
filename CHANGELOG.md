# Changelog — Krea 2 Edit Advanced

This fork continues the version numbers of [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit),
which ended at v2.0.0. Everything from "Before 3.0.0" down is the history of the original project plus one
change made in this fork before 3.0.0, and uses the original node names. LoRA weights:
https://huggingface.co/conradlocke/krea2-identity-edit

## 3.0.0 — Krea2 Edit Advanced: source hold and edit mask (breaking)

The package is renamed to `comfyui-krea2edit-advanced`. The nodes moved to the V3 schema under new ids, so they
can be installed next to the original pack: `Krea2EditAdvanced` ("Krea2 Edit Advanced (source patch)") replaces
`Krea2EditModelPatch`, and `Krea2EditAdvancedGroundedEncode` ("Krea2 Edit Advanced (grounded encode)") replaces
`Krea2EditGroundedEncode`. Workflows saved with the original nodes must be reconnected.

### Added
- **Source hold** (`source_hold`, `hold_strength`, `edit_mask`, `grow_percent`, `feather_percent`, available under
  `ref_position_a = aligned (trained)`): pulls the picture back to the source while it is drawn, so edits no
  longer shift or darken the rest of the picture, also over chains of edits. The edit mask limits where the
  picture may change; the grown (or shrunk) and feathered mask is returned as a `MASK` output.
- A warning when `source_latent_a` has another size than the sampled latent.

### Changed
- `ref_position_a` is a dynamic combo: the hold inputs appear under `aligned (trained)`. The other options show
  no extra inputs. `ref_boost_a/b`, `ref_boost_mask_a/b`, `ref_position_b` and the grounded encode inputs keep
  their meaning.
- Every input has a description in plain words, with the area it acts on.
- The example workflow is replaced by two examples: `krea2_edit_advanced_api.json` (one edit) and
  `krea2_edit_advanced_chain_api.json` (six edits with the latent pass). Both scale and center-crop the image to the
  output size before VAE Encode and use the default boosts (1.0).

### Removed
- The pixel path: `vae`, `source_image_a`, `source_image_b`, `target_latent` and `fit_mode`. The source is a
  latent only; scale the image to the output size before VAE Encode.
- The error for workflows that still use the pre-`_a`/`_b` input names.
- `node.zip` (a stale packaged copy of the original nodes), the original's funding file and its example workflow
  `krea2_identity_edit.json`.

## Before 3.0.0 (this fork, original node names) — consistent `_a` / `_b` input naming (breaking for saved workflows)

Every per-reference input now has an `_a` (first reference: the image being edited / the scene,
RoPE frame 1) and a `_b` (second reference: the person, RoPE frame 2) form, and each suffix always
acts on its own reference. Controls no longer change owner depending on how many references are connected.

### Changed
- `Krea2EditModelPatch`: `source_latent` → `source_latent_a`, `source_image` → `source_image_a`,
  `ref_boost` → `ref_boost_a`, `ref_boost_a` → `ref_boost_b`, `ref_position` → `ref_position_a`,
  `ref_position_a` → `ref_position_b`, `ref_boost_mask` → `ref_boost_mask_a`.
  **Single-image workflows keep their values** (the old unsuffixed controls drove the only image, now `_a`).
  **Two-image workflows must re-check the person's values:** the old unsuffixed controls drove the
  person and are now `_b` — see the table below.
- `Krea2EditGroundedEncode`: `image` → `image_a`.
- A workflow that still sends an old input name (`source_latent`, `source_image`, `image`, `ref_boost`,
  `ref_boost_mask`, `ref_position`) now fails with an error naming the new input, instead of running
  with the references silently dropped.
- A `_b` control without a `_b` reference is simply ignored (an unused `ref_boost_b` no longer builds an attention bias).
- Widget order is `ref_boost_a`, `ref_boost_b`, `fit_mode`, `ref_position_a`, `ref_position_b`, so saved
  single-image workflows keep their boost and position values on the only image.

### Added
- `source_latent_a` is now optional when `vae` + `source_image_a` are connected, so the pixel path no
  longer needs a separately VAE-encoded latent. In that mode the latent inputs are not read at all
  (previously they were still passed through `process_latent_in` and discarded). With neither
  connected the node raises a clear error.
- `ref_boost_mask_b`: a boost region for the second reference (previously the one mask could only target the last reference).

### Fixed
- **Boost masks now follow the reference's crop/resize.** The mask used to be squeezed onto the
  reference token grid as-is, but the reference had already been center-cropped or letterboxed
  to the target, so the boost landed on the wrong content whenever the source's aspect ratio
  differed from the output (most visible with `crop (legacy)` and on the latent path, ~15 px with
  `fit`). The mask is now cropped to the same window as its image (or latent) before use, and may
  have any resolution.
- A 2D `ref_boost_mask` was mis-shaped before resampling.

### Changed (behavior)
- **Boost masks are proportional.** Previously a mask was thresholded at 0.5 per token (all or
  nothing). A token whose mask value is `m` now gets the attention weight `1 + m × (boost − 1)`,
  so soft/feathered masks blend the boost in and gray below 0.5 is no longer dropped. Binary
  masks behave as before.

| Before (two references) | Now |
|---|---|
| `ref_boost` (person) | `ref_boost_b` |
| `ref_boost_a` (scene) | `ref_boost_a` |
| `ref_position` (person) | `ref_position_b` |
| `ref_position_a` (scene) | `ref_position_a` |
| `ref_boost_mask` (person) | `ref_boost_mask_b` |

Renamed sockets lose their links in workflows saved with the old names; reconnect them. The shipped
workflow of that time set both boosts to 4.0; the current one uses the defaults.

## v2.0.0 — Krea 2 Identity Edit V2 + Krea 2 General Edit V2

V2 ships as two LoRAs (weights on Hugging Face):
- `krea2_identity_edit_v2.safetensors` — **Krea 2 Identity Edit V2**: restage a person from one
  photo into any scene, pose, outfit and lighting. Extremely improved likeness, better sharpness
  and textures.
- `krea2_general_edit_v2.safetensors` — **Krea 2 General Edit V2**: instruction-based editing
  (remove, recolor, relight, backgrounds, add objects, text, outfits, restyles, flat lays).

Use one LoRA per task; loading both at 1.0 breaks the image (~0.6 each if combined).

### Added
- **Reference placement** on `Krea2EditModelPatch` — two new inputs, appended last so saved
  workflows keep their links; defaults reproduce the previous behavior exactly:
  - `ref_position` places the **last** reference (the person in two-image workflows, or the only image).
  - `ref_position_a` places the **first** reference (the scene).
  - Options: `aligned (trained)` (default; the reference is pinned onto the output layout),
    `beside (right of target)` (the reference sits next to the canvas in RoPE space — identity
    still flows, the layout pin is gone, so the person restages more freely), and the experimental
    `bag of features` (all tokens of a reference share one position).
  - Recommended for compositing a person into a scene: scene `aligned` + person `beside`.

## v1.2.5 — 2026-07-29

### Added
- **`target_latent` input on the patch node** (#15 — thanks @ethanfel, first outside
  code contribution!): wire the same latent that feeds `KSampler.latent_image` and the
  pixel path VAE-encodes the source at node-execution time instead of on the first
  sampling step. Without it, that mid-sampling encode can make ComfyUI evict part of
  the resident diffusion model on VRAM-tight setups — every remaining step then
  streams weights from CPU. Optional; unwired behavior is unchanged. See the README
  section "Pixel path and VRAM".

### Fixed
- Pre-encode console messages now report resolutions in pixels, not latent units.

## v1.2.4 — 2026-07-29

Node-only update (no new weights) — recommended for everyone on v1.2 weights.

### Fixed
- **Vertical-outpaint "doubling" band largely fixed.** Two geometry defects made the
  reference band misregister against the target grid: content was squashed by up to
  15px by /16 flooring, and odd-gap refs sat 8px off their true center from integer
  offset flooring. Sources are now center-cropped so the fitted axis lands on the /16
  grid exactly, and RoPE offsets are placed at the true (fractional) center. Biggest
  effect on outpaint, style-guard and removal edits. Note: the v1.2 weights learned
  some of this artifact during training, so rare cases can persist — `fit_mode: crop`
  remains a workaround; a fully clean from-scratch training is in progress for v2.
- From v1.2.3 (pushed, never release-tagged): a newer workflow meeting an older node
  install now warns "update the node pack" instead of crashing with a TypeError (#5);
  README `grounding_px` trained range corrected to the v1.2 reality (384–768).

### Advisory
- **Prefer `euler` (or other ODE samplers) over `er_sde` for outpainting.** The SDE
  noise injection disrupts the reference-copy channel and ruins outpaint coherence.

## v1.2 — 2026-07-17 (recommended)

`krea2_identity_edit_v1_2.safetensors` — pair it with the v1.2 nodes in this repo.

### New

**In the model:**
- **Better face likeness** on restaged subjects.
- **Character reference sheets** — both *using* a sheet as reference and *creating*
  one from a character.
- **Head / face swap** (and eye / person replacement).
- **Outpainting.**
- **Inpainting.**
- **Try-on** — put a garment onto a person.
- **Better person removal.**
- **Higher fidelity across the board, from a 1024 pass** — v1.1 had no high-resolution
  adaptation; v1.2 does.

**In the nodes:**
- **`ref_boost` — a reference-fidelity dial.** Turn how hard an edit locks onto the
  reference's appearance up or down (1.0 = neutral, >1 = pull harder toward the
  reference). Best value is model-specific.
- **No more blurry/stretched results (new `fit` geometry).** Sources are resampled to
  the target grid at a training-matched offset — and the old "match the source aspect
  ratio" requirement is gone. Needs `vae` + `source_image` connected on the patch node.

### Thanks
Head / face / eye / person swap is trained on **stablellama**'s MIT-licensed
[`change_eye_face_head_person`](https://huggingface.co/datasets/stablellama/change_eye_face_head_person)
dataset — big thanks for making it available.

### Node changes (technical)
- `fit_mode` defaults to `fit` (training-matched); `crop` remains for v1/v1.1-legacy weights.
- Added `ref_boost` / `ref_boost_a` reference-fidelity dials.

## v1.1 — 2026-07-09

`krea2_identity_edit_v1_1.safetensors`

- **Substantially improved face likeness and image fidelity**
- **Much stronger edit locality** — camera, pose, and untouched elements stay
  fixed far more reliably
- Better two-person identity separation
- More reliable object remove / replace
- Better compound outfit-change compliance
- Corrected reference geometry handling (training refs are now center-cropped,
  matching the shipped workflows)

**Known limitations of v1.1:**
- *Person*-replacement ("replace the woman with an orangutan") is currently
  weaker than v1 — keep v1 for that use case until v1.2
- No high-resolution adaptation pass yet: at high resolutions (especially
  two-person edits) identities can bleed together — prefer ~1–1.5MP and upscale
- `grounding_px`: v1.1's trained range is 384–768 (1024 often still works).
  If you get duplicated/split compositions, lower `grounding_px`.

## v1 — 2026-07-07

`krea2_identity_edit_v1.safetensors` — initial release. Remains available for
workflow reproducibility and for person-replacement edits.
