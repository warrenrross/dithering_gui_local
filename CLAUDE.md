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

1. `quantize_levels(values, bins)` — snaps float pixel values to the nearest of `bins` evenly-spaced levels between 0–255 using NumPy broadcasting.
2. `diffuse_channel(channel, algo_name, bins)` — runs an error-diffusion pass over a single 2-D uint8 channel using a pixel-by-pixel Python loop. **This is the performance bottleneck** — see Known Issues below.
3. `dither_image(image, grayscale, bins, algo_name)` — converts a PIL Image to grayscale or RGB, calls `diffuse_channel` per channel, returns a new PIL Image.

### UI layer (`DitherApp`)

**Controls panel (left column):**
- Three action buttons in workflow order: Open Image → Dither Image → Save Dithered As…
- Mode (Color / Greyscale), Bins spinbox, Algorithm combobox — none of these auto-trigger dithering.
- Zoom controls: `−` / entry field / `+` / Fit button. Entry accepts typed percentages (e.g. `150` or `150%`) or the word `Fit`. `+`/`−` step through `ZOOM_STEPS` presets.

**Preview panel (right column):**
- Two `tk.Canvas` widgets stacked vertically — Original on top, Dithered below — each taking equal vertical space via `rowconfigure(..., weight=1)`.
- A single vertical scrollbar (spans both canvas rows) and horizontal scrollbar at the bottom.
- Scrolling is synchronized: `canvas_orig` drives the scrollbars and syncs `canvas_proc` via `_yscroll_sync` / `_xscroll_sync`. Mousewheel works on either panel; Shift+scroll is horizontal.
- In Fit mode both canvases compute a shared display size (minimum of the two canvas dimensions) so both images are always at the same scale.

**Key design decision — load vs. dither separation:**
Opening an image only shows a preview. Dithering is never triggered automatically. The user must click "Dither Image" after choosing options. This prevents accidentally running a slow computation on the wrong image.

### Algorithms

Declared in the `ALGORITHMS` dict at module top. Each entry has a `divisor` (int) and `offsets` list of `(dx, dy, weight)` tuples. Adding a new algorithm = adding one dict entry, no other changes needed.

### Zoom state

`_zoom_factor` (float or None):
- `None` = Fit mode (auto-scale to canvas size)
- Float = fixed scale factor (e.g. `1.5` = 150%)

`ZOOM_STEPS` = preset list used by `+`/`−` buttons. Manual entry bypasses this and sets any value in [0.01, 32.0].

---

## Session history

### 2026-04-28 — Major UX and architecture overhaul

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
- `dither_fix_guide_v1.md` — Perplexity-generated guide covering vectorization, threading, and debounce patterns (not yet implemented)
- `dev_log.md` — detailed session change log

---

## Known issues / next tasks

### Performance (blocking UI)
`diffuse_channel` is a pure Python `for y / for x` double loop. On large images it blocks the main Tkinter thread, freezing the UI. Three fixes are documented in `dither_fix_guide_v1.md`:

1. **Vectorize with NumPy** — replace the Python loop with scanline-based NumPy operations. Reduces processing time from minutes to under a second for typical photos.
2. **Background thread** — run dithering in `threading.Thread` with `queue.Queue` to keep the UI responsive and allow cancellation.
3. **Debounce resize events** — `<Configure>` fires on every window move/resize. Use `after()` / `after_cancel()` so `_refresh_panels` only runs once after the user stops dragging.

### Minor
- Horizontal scroll sync is not tested under rapid mousewheel input — may drift between the two canvases.
- No progress indicator while dithering runs (blocked UI gives no feedback).
