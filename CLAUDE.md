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

1. `diffuse_channel_exact(channel, algo_name, bins, cancel_event, progress_fn)` — **primary path**. Uses a Numba `@njit(cache=True, nogil=True)` inner function `_process_row_exact` that runs the full sequential per-pixel loop in native code. Exact same-row (dy=0) propagation. Falls back to `diffuse_channel_fast` when Numba is not installed. Returns `None` if cancelled.
2. `diffuse_channel_fast(channel, algo_name, bins, cancel_event, progress_fn)` — **NumPy fallback**. Scanline-based: outer `for y` in Python, NumPy slices for inter-row propagation. Skips dy=0 offsets (approximation, but gives clean output). Kept as the no-Numba fallback.
3. `dither_image(image, grayscale, bins, algo_name, cancel_event, progress_fn)` — top-level entry point. Grayscale: calls `diffuse_channel_exact` on one channel. Color: submits R, G, B to `ThreadPoolExecutor(max_workers=3)` for parallel execution; uses a `threading.Lock`-protected shared row counter for thread-safe progress reporting.
4. `_prewarm_numba()` — called in a daemon thread at app startup to trigger Numba JIT compilation before the user's first click.

**Key design decision — separate buf/out arrays (applies to both functions):**
An earlier guide (and many online FS examples) writes same-row error propagation back into already-quantized values in a single array. This creates intermediate pixel values (e.g. 28, 199) instead of the correct quantized levels (0, 255 for bins=2). The fix is separate `buf` (pre-quantization accumulator) and `out` (final quantized output) arrays. In `diffuse_channel_fast`, dy=0 offsets are skipped entirely. In `diffuse_channel_exact`, the Numba inner loop handles dy=0 correctly because it processes pixels sequentially before any have been written to `out`.

**Key design decision — `nogil=True` is required for real parallelism:**
Numba's `@njit` holds the Python GIL by default. With `nogil=True`, threads genuinely run on separate CPU cores. Without it, the three channel threads queue up and parallel execution provides no speedup over sequential.

**Numba dependency:** Optional. `pip install numba` enables exact mode. Without it the app uses the NumPy fallback silently. The status bar shows `exact (Numba)` or `fast (install numba for exact)` at startup.

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
`run_dither` spawns a daemon `threading.Thread` (`_dither_worker`). The worker calls `dither_image` with a `cancel_event` (checked each scanline) and a `progress_fn` that puts `("progress", pct)` into a `queue.Queue`. `_poll_queue` runs every 50ms via `root.after` and drains the queue; terminal messages are `"done"`, `"cancelled"`, and `"error"`. All Tkinter widget calls happen only in `_poll_queue` (main thread).

For color dithering, `dither_image` itself uses `ThreadPoolExecutor(max_workers=3)` inside the worker thread — so there are 4 threads total: the Tkinter thread, the `_dither_worker` thread, and 3 channel executor threads. Progress from the executor threads is serialized via a `threading.Lock` before reaching the queue.

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

### 2026-04-28 (session 3) — Exact diffusion (Numba) + parallel R/G/B channels

**What changed:**

**`diffuse_channel_exact` (new primary path):**
- `@numba.njit(cache=True, nogil=True)` compiles `_process_row_exact` to native code
- Full sequential per-pixel loop inside Numba = exact same-row (dy=0) propagation
- Resolves the approximation in `diffuse_channel_fast`; bins=2 output is strictly binary
- `_prewarm_numba()` runs in a daemon thread at startup so first user click doesn't hit JIT delay
- Graceful fallback to `diffuse_channel_fast` when Numba is not installed
- Key lesson: `nogil=True` is mandatory for real parallelism — without it threads queue up and there's no speedup

**`dither_image` color branch rewritten to use `ThreadPoolExecutor(max_workers=3)`:**
- R, G, B channels submitted simultaneously; with `nogil=True` they run on separate cores
- Thread-safe progress via `threading.Lock`-protected shared row counter
- `cancel_event` shared across all futures; Cancel stops all three within one scanline

**Benchmarks:**

| Workload | Before | After |
|---|---|---|
| 512×512 single channel (exact) | ~3ms (approx) | ~3.7ms (exact) |
| 512×512 RGB parallel | ~11ms sequential | **~6.3ms** (~1.7× speedup) |
| 2000×2000 RGB parallel | ~220ms sequential | **~79ms** (~2.8× speedup) |

**Dependencies added:** `numba` (optional — `pip install numba`)

**Commit:** `1d7bc2a`

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

### Minor
- Horizontal scroll sync is not stress-tested under rapid mousewheel input — may drift between the two canvases.
- No serpentine (alternating-direction) scanning — currently always left-to-right. This is a standard dithering quality improvement that reduces directional banding.
- Parallel speedup sub-linear (~2.8× for 3 channels on 2000×2000) due to shared memory bandwidth between the three channel threads. Diminishing returns on images larger than ~4000px wide where RAM becomes the bottleneck.
- Numba JIT cold-start: first launch of the app after a clean `__pycache__` deletion will recompile. `cache=True` persists the compiled binary so subsequent launches are instant.

### Potential next features
- Serpentine scan order (alternating row direction per scanline — reduces the left-to-right banding artifact visible in large flat regions)
- Additional algorithms (Atkinson, Sierra, etc.) — one dict entry each in `ALGORITHMS`, no other changes needed
- Live preview: debounced auto-dither on option change (now safe since dithering is fast enough)
- Progress bar shows overall % but not per-channel breakdown — could show three mini-bars for R/G/B
