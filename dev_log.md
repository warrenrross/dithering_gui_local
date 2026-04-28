# Dithering App — Development Log

## Session: 2026-04-28

### Problems identified at start of session

- App froze the computer on load because dithering ran automatically every time an image was opened, a control was changed, or the window was resized. The core dithering algorithm (`diffuse_channel`) uses a pure Python pixel-by-pixel double loop — slow enough on large images to block the main Tkinter thread for minutes and starve the OS.
- No way to preview an image before committing compute to dithering it.

---

### Changes made

#### 1. Separated image loading from dithering
- Opening an image now only displays a preview of the original. No dithering happens until the user explicitly clicks the button.
- Removed all auto-trigger callbacks from the Mode radio buttons, Bins spinbox, and Algorithm combobox. Changing a setting no longer fires the dithering loop.

#### 2. Button order resequenced for logical workflow
- Previous order: Open Image → Save Dithered As → (options) → Apply/Refresh
- New order: Open Image → Dither Image → Save Dithered As → (options below)
- Renamed "Apply / Refresh" to "Dither Image" to make the action explicit.

#### 3. Replaced image Label widgets with Canvas widgets
- Switched both preview panels from `ttk.Label` to `tk.Canvas` to support scrolling and proper zoom.
- Added a shared vertical scrollbar and horizontal scrollbar that control both canvases simultaneously (synchronized scroll).
- Both canvases scroll together — mousewheel works on either panel; Shift+scroll goes horizontal.

#### 4. Added zoom controls
- Controls: `−` button, editable zoom entry field, `+` button, `Fit` button.
- `+` / `−` step through preset levels: 25% → 33% → 50% → 67% → 75% → 100% → 125% → 150% → 200% → 300% → 400%.
- Entry field accepts any typed value (e.g. `73` or `73%`); press Enter or click away to apply. Type `Fit` to return to auto-fit mode.
- Invalid input is silently ignored and the display resets to the last valid value.
- In Fit mode both panels use the same computed scale so they always display at identical size.

#### 5. Changed preview layout from side-by-side to top/bottom
- Original image panel on top, Dithered image panel below, each taking equal vertical space.
- Single scrollbar column shared by both panels.

#### 6. Fixed Fit mode bug
- `_refresh_panels` had `min(canvas.winfo_width(), other.winfo_width(), 1)` — the literal `1` was being passed as a third argument to `min()`, so the canvas size was always evaluated as 1px and then clamped to 100px. Images were being fit into a 100×100 box regardless of actual window size.
- Fixed by computing `min(w1, w2)` separately and then applying the 100px floor with `max(..., 100)`.

#### 7. Fixed Pillow deprecation warning
- `Image.fromarray(arr, mode="RGB")` and `Image.fromarray(arr, mode="L")` were using a deprecated keyword argument.
- Changed to `Image.fromarray(arr)` — Pillow infers mode from the array shape and dtype automatically.

---

### Known remaining issue

The core dithering loop (`diffuse_channel` in `dither_gui.py`) is still a pure Python pixel-by-pixel double `for` loop. On large images this will still block the UI thread and take a long time. The reference document `dither_fix_guide_v1.md` covers the recommended fixes:
1. Vectorize the loop using NumPy (eliminates the performance problem).
2. Run dithering in a background thread with `threading.Thread` + `queue.Queue` (keeps UI responsive during processing).
3. Debounce the `<Configure>` resize event with Tkinter's `after()` / `after_cancel()` (reduces redundant redraws on window drag).
