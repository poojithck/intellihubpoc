## Optional Tiling ("Crop-Then-Ask") Design

This document proposes an optional, per-SOR tiling strategy to improve small-object recognition and text legibility while preserving whole-scene reasoning where needed. It is designed to plug into the current pipeline with minimal disruption and clear switches to enable/disable per SOR.

### Goals
- Improve accuracy on small targets (e.g., fuse barrels, SPD/MPD, NMI/date on asbestos bags) without breaking whole-board tasks.
- Respect AWS Bedrock 5MB per image (encoded) limit and current multi-image input handling.
- Avoid double counting when objects span tile boundaries.
- Keep backward compatibility; default behavior unchanged unless explicitly enabled.

### Non-Goals
- No immediate model/prompt rewrites across all SORs. Only additive, opt-in tiling support.
- No heavy CV pre-processing (e.g., classical detectors). We rely on the LLM and simple tiling logic.

## Where tiling helps vs. not recommended

### Recommended for tiling (primary or assistive)
- FuseReplacement: small cylindrical barrels; local detail matters.
- ServiceProtectionDevices: small black/translucent protection boxes; easy to miss at overview scale.
- AsbestosBagAndBoard: reading NMI digits/date from the bag; tiles improve OCR-like recognition.
- Legacy meter_reading (if used): tiles around register improve digit reading.

### Prefer overview (tiling optional as verification only)
- MeterConsolidationE4: requires full-board reasoning; use overview(s) as primary; tiles optional to confirm ambiguous items.
- PlugInMeterRemoval: board-level count/compare; overview primary; tiles optional to confirm presence/absence on black plates.
- NeutralLinkInstallation, SwitchInstallation: primarily board-level; tiles optional for confirmatory evidence.

## Constraints and existing building blocks

- BedrockClient:
  - `invoke_model_multi_image(prompt, images, ...)` allows multi-image per request, injecting a metadata text line after each image with the image name (and timestamp if present).
  - Hard check for <= 5MB encoded per image.
- ImageLoader:
  - Centralized resize/encode with bulletproof size limiting (`encode_single_image`) honoring 5MB cap.
  - EXIF orientation and timestamp extraction already handled.
- UnifiedSORProcessor:
  - Targeted mode (per-folder image selection), plus legacy grids path.
  - Per-SOR prompts preloaded and multi-image calls already implemented.
- ImageGridder: Builds grids (multi-photo collages). Tiling will be a separate path (not a replacement).

## High-level architecture

1) Config-driven, opt-in tiling per SOR/type.
2) Two supported modes:
   - Hybrid: 1 downscaled overview image + N high-res tiles.
   - ROI-first two-pass (optional upgrade): first pass returns bounding boxes from an overview; second pass crops exact ROIs.
3) Deduplication strategies to avoid double counts across overlapping tiles.
4) Prompt augmentation only when tiling is active to guide counting/aggregation.
5) Batching/chunking logic to cap images per request and costs.

## Configuration (proposal)

Add to `configs/app_config.yaml` under `image_processing`:

```yaml
image_processing:
  default_resize: { width: 900, height: 900 }
  default_format: PNG
  crop_then_ask:
    enabled: false                 # master switch
    mode: hybrid                   # hybrid | roi_first
    tile_size: [512, 512]          # width, height in px
    overlap_ratio: 0.15            # 15% overlap (hybrid)
    safe_zone_ratio: 0.15          # 15% inset for counting rule (hybrid)
    include_overview: true         # always include 1 overview for context
    overview_max_dim: 900          # max side length for overview (px)
    max_tiles_per_image: 12        # cap tiles per original image
    max_tiles_per_request: 12      # cap tiles per API request
    preferred_tile_format: JPEG    # JPEG to stay under 5MB with photos
    sor_overrides:                 # enable per SOR with optional overrides
      FuseReplacement:
        enabled: true
        mode: hybrid
      ServiceProtectionDevices:
        enabled: true
        mode: hybrid
      AsbestosBagAndBoard:
        enabled: true
        mode: hybrid
      MeterConsolidationE4:
        enabled: false             # overview only
      PlugInMeterRemoval:
        enabled: false             # overview primary; tiles optional later
      NeutralLinkInstallation:
        enabled: false
      SwitchInstallation:
        enabled: false
```

Notes:
- We keep defaults conservative; only a subset enabled by default.
- These settings are read via `ConfigManager.get_image_processing_config()` and passed to the tiling logic at prepare time.

## New module: `src/tools/image_tiler.py` (planned)

Responsibilities:
- Generate tiles for a PIL image given `tile_size` and `overlap_ratio`.
- Return `(tile_name, tile_image, tile_meta)` tuples, where `tile_meta` includes parent image name and tile coordinates (x, y, w, h) in pixels.
- Support an optional “safe zone” ratio for center-based dedup rules (used in prompts; dedup post-processing is handled in aggregation).

API sketch:
```python
class ImageTiler:
  def __init__(self, tile_w: int, tile_h: int, overlap_ratio: float):
    ...

  def tile(self, img: Image.Image, name: str) -> List[Tuple[str, Image.Image, Dict]]:
    # returns [("{name}_rX_cY.png", tile_img, {"x": int, "y": int, "w": int, "h": int, "row": int, "col": int, "parent": name})]
```

Implementation notes:
- Use stride = `tile_dim * (1 - overlap_ratio)`, clamp at image boundaries.
- Do not upscale small images; skip tiling if the image is smaller than `tile_size`.

## Integration points

### 1) Targeted image preparation (preferred)
File: `scripts/unified_sor_processor.py`, function: internal `prepare_images_sync` (targeted branch).

Add a tiling branch:
1. Read tiling config via `ConfigManager.get_image_processing_config()`.
2. For each SOR being analyzed:
   - If `sor_overrides[SOR].enabled` (or global `enabled`) is true, build inputs using tiling.
   - Otherwise, keep existing behavior.
3. Tiling build (hybrid mode):
   - For each selected original image:
     - Add 1 overview (copy → `thumbnail((overview_max_dim, overview_max_dim))`, encode via `ImageLoader.encode_single_image`, prefer JPEG; name suffix: `_overview`)
     - Generate tiles with `ImageTiler`.
     - Trim to `max_tiles_per_image`.
     - Encode each tile (JPEG) with `ImageLoader.encode_single_image`.
     - Name tiles with coordinate info so Bedrock metadata line includes it, e.g., `orig_r{row}_c{col}_x{x}_y{y}_w{w}_h{h}.jpg`.
   - Combine all overview+tiles across images, then cap per-request with `max_tiles_per_request` (chunk if needed).
4. Media type: set `media_type="image/jpeg"` on encoded items.

### 2) Legacy grids path (optional)
File: `src/tools/image_analyzer.py` + `ImageGridder`

Keep as-is. Tiling is an alternative preparation path, not a replacement for grids.

## Prompt augmentation (only when tiling is active)

Add a short “tiling addendum” appended to each SOR `main_prompt` at runtime when tiling is enabled for that SOR. Implement this in `UnifiedSORProcessor._analyze_sor_type` when composing `prompt_text`.

Suggested addendum (hybrid mode):

- For counting tasks (e.g., FuseReplacement, SPD/MPD):
  - “You will receive one low-res overview plus multiple tiles that include tile filenames and coordinates. Count unique objects across all tiles. Use the ‘center-in-safe-zone’ rule: only count an object in a tile if its visual center lies within the inner safe zone (≈<safe_zone_ratio> inset from each tile edge). Ignore any object clipped by tile edges; it will be counted in a neighboring tile. Provide the final unique count only. If needed, reference tile filenames in your notes.”

- For text extraction (AsbestosBagAndBoard):
  - “Use tiles to read small text (NMI/date). If multiple tiles contain the same text, de-duplicate; return a single value.”

Prompt injection mechanics:
- Build `prompt_text = f"{system}\n\n---\n\n{main}\n\n---\n\n{tiling_addendum}"` when tiling is active for that SOR.
- Keep the JSON schema unchanged.

## Request packaging & limits

- Per request cap (`max_tiles_per_request`): 8–12 is a good baseline. If more tiles are produced, issue multiple requests and aggregate.
- Encoding: use JPEG for tiles to remain well under 5MB even for 512–640px tiles.
- Overview size: ≤ 900 px on the long side (already used elsewhere).

## Aggregation & de-duplication

Approaches (choose per SOR):

1) Safe-zone rule (single-pass; simplest):
   - We ask the model to count only objects whose centers are within an inner rectangle (tile minus `safe_zone_ratio` margins). This discourages counting objects that straddle edges.
   - Because we do not receive per-object coordinates, duplication risk is reduced but not eliminated; however, with overlap and the rule, duplicates become rare in practice.

2) ROI-first two-pass (most robust):
   - Pass 1 (overview only): prompt the model to produce a structured list of ROI boxes (x, y, w, h in percent) per relevant object (e.g., fuse barrels). Limit max proposals per image.
   - Post-process: convert ROIs to pixel boxes; perform simple NMS/merge if overlaps are high.
   - Pass 2: crop ROIs and re-analyze tiles (or ask to validate/classify each ROI). Aggregate deterministically by ROI index.

3) Non-overlapping tiles with padded context (middle ground):
   - Crop non-overlapping tiles but render each tile with an internal padding margin (expand crop by ~10–15% clamped to image bounds) before resize, so edge objects are still singularly visible in a single tile.

Recommended initial path: Safe-zone rule for FuseReplacement & SPD; consider ROI-first if duplicates observed.

Aggregation mechanics in code:
- Keep per-request outputs per SOR as-is; JSON schema unchanged (e.g., counts and booleans).
- If multiple requests are needed (chunked tiles):
  - For boolean decisions (e.g., devices_added): OR semantics across requests (any positive wins), resolving ties by best-notes clarity.
  - For counts: take the max among requests when all are tile subsets of the same originals (safe-zone rule aims to avoid double counts). For ROI-first, sum unique ROIs.
  - For “which image” fields: prefer overview filenames for board-level fields, tile filenames only in notes for evidence.

## Naming & metadata conventions

- Tile filename: `"{orig}_{kind}_{row}-{col}_x{x}_y{y}_w{w}_h{h}.jpg"`, where `kind` ∈ {`overview`, `tile`}.
- Because `BedrockClient.invoke_model_multi_image` inserts a text line “Image i of n: {name}”, coordinate info becomes part of the model-visible context.
- If needed later, add a `"tile_bbox"` key in the image dict for downstream logging (ignored by Bedrock; kept for audit).

## Error handling, logging, and metrics

- Log per-image encoded size in MB (already done in `ImageGridder.encode_grids`). Mirror this for tiling inputs.
- Track number of tiles generated per original and how many were sent after capping.
- Record request batches per SOR and per work order; include costs and token counts (already captured by `BedrockClient`).
- Add a run-time flag to dump a contact sheet (debug only): draw tile boxes on the original and save to `artefacts/test_tiles/` for spot checks.

## Testing strategy

1) Unit scope:
   - `ImageTiler.tile` correctness (coverage of edges, overlap stride, no upscaling; coordinate accuracy).
   - Encoding constraints: all tiles ≤ 5MB encoded; overview ≤ 5MB.

2) Integration scope:
   - For FuseReplacement and SPD, compare baseline vs. tiling on a small gold set; expect improved recall on small objects while maintaining precision.
   - Verify no obvious duplicate counts on border-straddling objects (inspect failure cases; adjust `safe_zone_ratio` and overlap if needed).

3) Regression scope:
   - Ensure SORs with tiling disabled remain unchanged in outputs.
   - Verify performance impact (wall time, total tokens) remains acceptable; adjust caps if needed.

Acceptance criteria:
- FuseReplacement: +X% accuracy on count correctness (target +5–10%); stable false-positive rate.
- SPD/MPD: +recall on added devices with minimal precision loss.
- AsbestosBagAndBoard: improved NMI/date extraction on small handwriting photos.

## Rollout plan

1) Phase 1 (behind a config flag): enable tiling for FuseReplacement and SPD on a small subset (test mode / developer run).
2) Phase 2: expand coverage to AsbestosBagAndBoard; monitor token costs and latency.
3) Phase 3: consider ROI-first for the SORs where duplicates remain problematic.

## Risks and mitigations

- Duplicate counts across tiles:
  - Mitigate with `safe_zone_ratio`; if persisting, switch to ROI-first for that SOR.
- Cost/time increase due to more images per request:
  - Cap tiles per image and per request; keep JPEG; reduce overlap/tile size if needed.
- Loss of global context:
  - Always include an overview in hybrid mode; avoid tiling as primary for board-level SORs.
- Inconsistent prompts across versions:
  - Inject addendum dynamically at runtime only when tiling enabled; leave stored YAML intact.

## Work items (implementation checklist)

- Config
  - [ ] Add `image_processing.crop_then_ask` to `configs/app_config.yaml` with defaults and per-SOR overrides.

- Tiler
  - [ ] Create `src/tools/image_tiler.py` with `ImageTiler` (tile/overlap/coords) and tests.

- Integration
  - [ ] In `UnifiedSORProcessor.prepare_images_sync` (targeted branch), add tiling pipeline gated by config.
  - [ ] Encode overview+tiles via `ImageLoader.encode_single_image` (JPEG), inject `media_type`.
  - [ ] Chunk images if > `max_tiles_per_request` and aggregate results.

- Prompt augmentation
  - [ ] In `_analyze_sor_type`, append a small tiling addendum to `prompt_text` only when tiling is active for that SOR (values pulled from config, e.g., safe-zone percent).

- Aggregation utilities
  - [ ] Implement simple aggregators per SOR interface (e.g., union-of-booleans, max-of-counts, ROI-first merge if enabled later).
  - [ ] Optional: debug contact sheets with tile boxes for QA.

- Monitoring
  - [ ] Add logs: tiles generated/sent per image, encoded MB, per-request image counts.

- Documentation
  - [ ] Update `README.md` usage section to explain tiling flags and when to enable per SOR.

## Future extensions

- ROI-first two-pass prompts stored as separate YAMLs (e.g., `FuseReplacement_ROI.yaml`) to keep concerns separated from decision prompts.
- Lightweight visual saliency pass (single small model) to propose ROIs quickly before invoking Bedrock.
- Learned tile selection based on uncertainty feedback.
- Adaptive tile size: start coarse, refine where the model reports uncertainty.

---

This plan keeps the pipeline modular, opt-in, and low-risk. It leverages existing multi-image support, preserves current behavior by default, and provides clear hooks to evaluate and gradually adopt tiling where it meaningfully improves accuracy.


