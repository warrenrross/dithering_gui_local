# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Running the app

```bash
pip install pillow numpy
python3 dither_gui.py
```

Tkinter is required and ships with most standard Python installs.
Use `python3`, not `python` — `python` is not on PATH in this environment.

## Architecture

The entire application lives in `dither_gui.py` — no package structure.

### Core pipeline

1. `diffuse_channel_fast(channel, algo_name, bins, cancel_event, progress_fn)` — scanline-based NumPy error diffusion over a single 2-D uint8 channel. Uses two arrays: `buf` accumulates inter-row errors; `out` stores clean quantized values. Same-row (dy=0) kernel offsets are skipped to prevent corrupting already-quantized output pixels. Returns `None` if `cancel_event` is set.
2. `dither_image(image, grayscale, bins, algo_name, cancel_event, progress_fn)` — converts a PIL Image to grayscale or RGB, calls `diffuse_channel_fast` per channel, returns a new PIL Image or `None` if cancelled.

**Key design decision — separate buf/out arrays:**
An earlier guide (and many online FS examples) writes same-row error propagation back into already-quantized values in a single array. This creates intermediate pixel values (e.g. 28, 199) instead of the correct quantized levels (0, 255 for bins=2). The fix is separate `buf` (pre-quantization accumulator) and `out` (final quantized output) arrays, with dy=0 offsets skipped.

### UI layer (`DitherApp`)

**Controls panel (left column):**
- Action buttons in workflow order: Open Image → Dither Image → Cancel → (progress bar) → Save Dithered As…
- Cancel is disabled until dithering starts; Open and Dither disable while a job is running.
- Progress bar (0–100%) fills as scanlines complete; resets to 0 on cancel/error.
- Mode (Color / Greyscale), Bins spinbox, Algorithm combobox — none of these auto-trigger dithering.
- Zoom controls: `−` / entry field / `+` / Fit button. Entry accepts typed percentages (e.g. `150` or `150%`) or the word `Fit`. `+`/`−` step through `ZOOM_STEPS` presets.

**Preview panel (right column):**
- Two `tk.Canvas` widgets stacked vertically — Original on top, Dithered below — each taking equal vertical space via `rowconfigure(..., weight=1)`.
- A single vertical scrollbar (spans both canvas rows) and horizontal scrollbar at the bottom.
- Scrolling is synchronized: `canvas_orig` drives the scrollbars and syncs `canvas_proc` via `_yscroll_sync` / `_xscroll_sync`. Mousewheel works on either panel; Shift+scroll is horizontal.
- In Fit mode both canvases compute a shared display size (minimum of the two canvas dimensions) so both images are always at the same scale.

**Key design decision — load vs. dither separation:**
Opening an image only shows a preview. Dithering is never triggered automatically. The user must click "Dither Image" after choosing options. This prevents accidentally running a slow computation on the wrong image.

**Threading model:**
`run_dither` spawns a daemon `threading.Thread`. The worker calls `dither_image` with a `cancel_event` (checked each scanline) and a `progress_fn` that puts `("progress", pct)` into a `queue.Queue`. `_poll_queue` runs every 50ms via `root.after` and drains the queue; terminal messages are `"done"`, `"cancelled"`, and `"error"`. All Tkinter widget calls happen only in `_poll_queue` (main thread).

**Resize debounce:**
`_on_resize` restarts a 200ms `after()` timer on every `<Configure>` event. `_handle_resize_done` is called only once after dragging stops.

### Algorithms

Declared in the `ALGORITHMS` dict at module top. Each entry has a `divisor` (int) and `offsets` list of `(dx, dy, weight)` tuples. Adding a new algorithm = adding one dict entry, no other changes needed.

### Zoom state

`_zoom_factor` (float or None):
- `None` = Fit mode (auto-scale to canvas size)
- Float = fixed scale factor (e.g. `1.5` = 150%)

`ZOOM_STEPS` = preset list used by `+`/`−` buttons. Manual entry bypasses this and sets any value in [0.01, 32.0].

---

## Session history

### 2026-04-28 (session 1) — Major UX and architecture overhaul

**What changed:**
- Separated image loading from dithering (no auto-trigger)
- Reordered buttons to match workflow: Open → Dither → Save
- Replaced `ttk.Label` image panels with `tk.Canvas` panels (enables scrolling/zoom)
- Added synchronized scrolling between the two preview canvases
- Added zoom controls with editable entry field (arbitrary % or Fit)
- Changed preview layout from side-by-side to top/bottom stacking
- Fixed Fit mode bug: `min(..., 1)` was clamping canvas size to 100px
- Fixed Pillow deprecation: removed `mode=` keyword from `Image.fromarray()` calls

**Reference docs added this session:**
- `dither_fix_guide_v1.md` — Perplexity-generated guide covering vectorization, threading, and debounce patterns
- `dev_log.md` — detailed session change log

---

### 2026-04-28 (session 2) — Performance overhaul + Git setup

**What changed:**

**Git / repo:**
- Initialized git, added `.gitignore` (excludes `__pycache__`, `*.pyc`, `.DS_Store`, `*.dmg`, venv)
- Pushed to `https://github.com/warrenrross/dithering_gui_local` — two commits on `main`

**Core algorithm (`diffuse_channel_fast`):**
- Replaced `diffuse_channel` (double Python pixel loop) with scanline NumPy
- 512×512 single channel: ~minutes → **~3ms**
- Discovered a bug in the reference guide's approach: writing same-row (dy=0) errors back into already-quantized `arr[y]` creates intermediate pixel values instead of correct quantized levels. Fixed by using separate `buf`/`out` arrays and skipping dy=0 offsets
- `cancel_event` checked each scanline; `progress_fn(y, h)` callback for per-row progress

**Threading (`run_dither` → `_dither_worker` + `_poll_queue`):**
- Dithering runs in a daemon thread; results delivered via `queue.Queue`
- Progress bar (0–100%), Cancel button (stops within one scanline)
- Open/Dither buttons disabled while job runs; re-enabled on done/cancel/error
- All widget updates stay on the Tkinter main thread

**Resize debounce (`_on_resize` + `_handle_resize_done`):**
- 200ms `after()`/`after_cancel()` debounce; `_refresh_panels` fires once after drag stops

**Removed:** `quantize_levels()`, old `diffuse_channel()` (replaced by `diffuse_channel_fast`)

---

## Known issues / next tasks

### Same-row error propagation (approximation)
The scanline NumPy implementation skips dy=0 kernel offsets (right-neighbor errors within the same row) to avoid corrupting quantized output. For most real images the visual difference from a fully sequential implementation is subtle. For pathological inputs (large flat regions at bins=2) the dithering pattern differs from a pixel-by-pixel reference. If exact same-row propagation is needed, the next step is Numba or Cython — there is no pure NumPy solution without a sequential inner loop.

### Minor
- Horizontal scroll sync is not stress-tested under rapid mousewheel input — may drift between the two canvases.
- Color dithering processes R, G, B channels sequentially; a 3-channel image takes ~3× as long as grayscale. Could parallelize with `concurrent.futures.ThreadPoolExecutor` (GIL releases during NumPy ops).
- No serpentine (alternating-direction) scanning — currently always left-to-right. This is a standard dithering quality improvement.

### Potential next features
- Serpentine scan order (alternating row direction, reduces directional banding)
- Parallel R/G/B channel processing with ThreadPoolExecutor
- Additional algorithms (Atkinson, Sierra, etc.) — one dict entry each in `ALGORITHMS`
- Live preview: debounced auto-dither on option change (now safe since dithering is fast)
