# Python Dithering GUI

Apply error-diffusion dithering algorithms to images with a live side-by-side comparison.

## Features

- Load any image (PNG, JPEG, BMP, GIF, TIFF, WebP)
- Preview original before committing to dithering
- Color or greyscale dithering
- Adjustable quantization bins / levels (2–64)
- Four algorithms: Floyd-Steinberg, Jarvis/Judice/Ninke, Stucki, Burkes
- **Exact error diffusion** with Numba JIT (install `numba` for correct same-row propagation; falls back to fast NumPy approximation without it)
- **Parallel R/G/B processing** — all three color channels dither simultaneously on separate CPU cores
- Non-blocking: dithering runs in a background thread; UI stays responsive
- Progress bar + Cancel button — stop any job within one scanline
- Status bar shows active mode: `exact (Numba)` or `fast`
- Stacked original / dithered preview panels with synchronized scrolling
- Zoom controls: preset steps via `+`/`−`, or type any percentage directly
- Save result as PNG or JPEG

## Workflow

1. **Open Image** — loads a preview; no processing yet
2. Select Mode, Bins, and Algorithm
3. **Dither Image** — runs in the background; progress bar shows completion
4. **Cancel** — stops the job at any time (within one scanline)
5. **Save Dithered As…** — export when satisfied

## Requirements

```bash
pip install pillow numpy          # required
pip install numba                 # optional — enables exact diffusion mode
```

Tkinter ships with most standard Python installs.

## Run

```bash
python3 dither_gui.py
```

## Zoom

| Action | Result |
|--------|--------|
| `+` / `−` buttons | Step through presets (25% → 33% → … → 400%) |
| Type a number + Enter | Jump to any zoom (e.g. `73` or `73%`) |
| Type `Fit` + Enter | Return to auto-fit |
| **Fit** button | Return to auto-fit |
| Scroll wheel | Pan vertically (both panels in sync) |
| Shift + scroll | Pan horizontally |
