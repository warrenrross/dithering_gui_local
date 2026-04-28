# Dithering App Fix Guide

This guide explains how to improve a local Python/Tkinter dithering app in three areas: the dithering core, background processing, and resize-event handling. Error-diffusion dithering works by quantizing a pixel and distributing the resulting error to future pixels according to a fixed kernel; Floyd–Steinberg and Jarvis–Judice–Ninke are standard examples of this family.[cite:1][cite:2][cite:22]

## Priority order

The highest-impact fix is the dithering core because the current implementation uses nested Python loops over pixels, which is exactly the expensive pattern that dominates runtime in error-diffusion code examples.[cite:18][cite:22] The second fix is moving the computation off the Tkinter main thread so the GUI remains responsive, because Tkinter's event loop must not be blocked by long-running work and queue-based polling is a standard way to hand results back safely.[cite:26][cite:29] The third fix is debouncing the `<Configure>` event, because Tkinter can emit many resize events during a drag and `after()` with `after_cancel()` is the standard way to coalesce them into one delayed refresh.[cite:24][cite:32]

## Core algorithm

### Reality check on full vectorization

Floyd–Steinberg and Jarvis–Judice–Ninke are causal error-diffusion algorithms: each output pixel depends on the quantization error produced by earlier pixels in scan order.[cite:1][cite:2][cite:22] That dependency chain means there is no simple fully vectorized NumPy expression that produces the exact same result as a left-to-right, top-to-bottom implementation; the practical improvement is to remove the inner `for x` loop and operate one full row at a time with NumPy slices, while keeping a scanline loop over rows.[cite:18][cite:19][cite:22]

### Required skills

- Python performance tuning, especially understanding where Python loops are expensive.
- NumPy array programming with slicing, temporary buffers, broadcasting, and clipping.
- Image representation decisions, including `float32` working buffers and `uint8` output conversion.
- Algorithm validation, because optimized error-diffusion code must be compared visually and numerically against a known-correct reference.[cite:18][cite:19][cite:22]

### Implementation spec

- Keep the image in a `float32` NumPy array during diffusion to avoid repeated type conversions and to preserve fractional propagated error.
- Quantize one scanline at a time using vectorized nearest-level or rounded-bin math.
- Compute a row-wide error array once per scanline, then distribute that error to the right neighbor and to the next one or two rows using slice updates.
- Process grayscale as a 2D array and RGB as a 3D array with the same scanline code operating over the last channel dimension.
- Clip only after the full diffusion pass, then convert back to `uint8`.
- Optionally add serpentine scanning later, but first preserve a single deterministic raster order to simplify testing.[cite:1][cite:2][cite:22]

### Working NumPy example

The following code keeps only a scanline loop in Python. Within each row, quantization and error propagation are done with NumPy slice operations. This is not mathematically "fully vectorized" across the entire image, but it removes the per-pixel Python loop and is the practical NumPy-only optimization path for exact raster-order error diffusion.[cite:18][cite:19][cite:22]

```python
import numpy as np


def quantize_row(row, bins):
    bins = max(2, int(bins))
    levels = bins - 1
    return np.round(row * levels) / levels


def fs_dither_scanlines(image, bins=2):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]

    arr = arr / 255.0
    h, w, c = arr.shape

    for y in range(h):
        old = arr[y].copy()
        new = quantize_row(old, bins)
        err = old - new
        arr[y] = new

        if w > 1:
            arr[y, 1:] += err[:-1] * (7.0 / 16.0)

        if y + 1 < h:
            arr[y + 1, :] += err * (5.0 / 16.0)
            arr[y + 1, 1:] += err[:-1] * (1.0 / 16.0)
            arr[y + 1, :-1] += err[1:] * (3.0 / 16.0)

    arr = np.clip(arr, 0.0, 1.0)
    out = (arr * 255.0 + 0.5).astype(np.uint8)
    return out[..., 0] if image.ndim == 2 else out


def jjn_dither_scanlines(image, bins=2):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]

    arr = arr / 255.0
    h, w, c = arr.shape

    for y in range(h):
        old = arr[y].copy()
        new = quantize_row(old, bins)
        err = old - new
        arr[y] = new

        if w > 1:
            arr[y, 1:] += err[:-1] * (7.0 / 48.0)
        if w > 2:
            arr[y, 2:] += err[:-2] * (5.0 / 48.0)

        if y + 1 < h:
            arr[y + 1, :] += err * (7.0 / 48.0)
            if w > 1:
                arr[y + 1, 1:] += err[:-1] * (5.0 / 48.0)
                arr[y + 1, :-1] += err[1:] * (5.0 / 48.0)
            if w > 2:
                arr[y + 1, 2:] += err[:-2] * (3.0 / 48.0)
                arr[y + 1, :-2] += err[2:] * (3.0 / 48.0)

        if y + 2 < h:
            arr[y + 2, :] += err * (5.0 / 48.0)
            if w > 1:
                arr[y + 2, 1:] += err[:-1] * (3.0 / 48.0)
                arr[y + 2, :-1] += err[1:] * (3.0 / 48.0)
            if w > 2:
                arr[y + 2, 2:] += err[:-2] * (1.0 / 48.0)
                arr[y + 2, :-2] += err[2:] * (1.0 / 48.0)

    arr = np.clip(arr, 0.0, 1.0)
    out = (arr * 255.0 + 0.5).astype(np.uint8)
    return out[..., 0] if image.ndim == 2 else out
```

### Integration steps

1. Convert the current per-channel pixel loop into a function that accepts a full NumPy array rather than a Pillow image directly.
2. Normalize to `[0, 1]`, dither with one of the scanline functions above, then convert back to a Pillow image only at the boundary of the GUI layer.
3. Keep the function pure: input array in, output array out, with no Tkinter or file I/O inside it.
4. Benchmark on a large image before and after the change, because the point is to reduce time spent in Python-level loops.[cite:18][cite:22]

### Limits and next options

A NumPy-only scanline implementation is the cleanest local fix, but exact error diffusion still remains sequential by scan order.[cite:18][cite:19] If more speed is needed after this refactor, the next practical step is JIT or native acceleration such as Numba, Cython, or a compiled extension rather than trying to force full-image vectorization where the dependency graph does not fit it well.[cite:19]

## Threading pattern

### Goal

Tkinter should stay on the main thread while the heavy dithering computation runs in a worker thread; the worker must never call Tk widgets directly, and the GUI should poll a `queue.Queue` for completion, progress, or error messages.[cite:26][cite:29] Cancellation should be cooperative, typically via a `threading.Event` checked by the worker between chunks of work.[cite:23][cite:26]

### Required skills

- Tkinter event-loop rules, especially that widget updates stay on the GUI thread.
- Python threading basics, including daemon threads, `threading.Event`, and safe shutdown.
- Producer/consumer communication with `queue.Queue`.
- Chunked algorithm design, because cancellation only works if the worker checks the cancel flag regularly.[cite:23][cite:26][cite:29]

### Implementation spec

- Start one worker thread per processing request.
- Pass immutable inputs or copied NumPy arrays into the worker.
- Use a queue message protocol such as `("progress", value)`, `("done", result)`, `("error", exc_text)`, and `("cancelled", None)`.
- Use `root.after(...)` polling every 50 to 100 ms to drain the queue on the main thread.
- Disable controls while work is running, re-enable them on `done`, `error`, or `cancelled`.
- Use a `threading.Event` for cancellation and check it inside the worker's scanline loop.[cite:23][cite:26][cite:29]

### Minimal working example

```python
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Threaded Tkinter Demo")

        self.q = queue.Queue()
        self.worker = None
        self.cancel_event = None
        self.poll_id = None

        self.status = tk.StringVar(value="Idle")
        self.progress = tk.IntVar(value=0)

        ttk.Button(root, text="Start", command=self.start_job).pack(padx=12, pady=6)
        ttk.Button(root, text="Cancel", command=self.cancel_job).pack(padx=12, pady=6)
        ttk.Label(root, textvariable=self.status).pack(padx=12, pady=6)
        ttk.Progressbar(root, maximum=100, variable=self.progress, length=260).pack(padx=12, pady=6)

    def start_job(self):
        if self.worker and self.worker.is_alive():
            return
        self.cancel_event = threading.Event()
        self.status.set("Running")
        self.progress.set(0)
        self.worker = threading.Thread(target=self.run_job, daemon=True)
        self.worker.start()
        self.poll_queue()

    def run_job(self):
        try:
            total = 100
            for i in range(total):
                if self.cancel_event.is_set():
                    self.q.put(("cancelled", None))
                    return
                time.sleep(0.03)
                self.q.put(("progress", i + 1))
            result = "Finished successfully"
            self.q.put(("done", result))
        except Exception as e:
            self.q.put(("error", str(e)))

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "progress":
                    self.progress.set(payload)
                    self.status.set(f"Running: {payload}%")
                elif kind == "done":
                    self.status.set(payload)
                    self.progress.set(100)
                    return
                elif kind == "cancelled":
                    self.status.set("Cancelled")
                    return
                elif kind == "error":
                    self.status.set(f"Error: {payload}")
                    return
        except queue.Empty:
            pass

        if self.worker and self.worker.is_alive():
            self.poll_id = self.root.after(50, self.poll_queue)
        else:
            self.poll_id = None

    def cancel_job(self):
        if self.cancel_event is not None:
            self.cancel_event.set()


root = tk.Tk()
App(root)
root.mainloop()
```

### How to apply it to the dithering app

Wrap the new scanline dithering function in a worker that processes one row at a time and checks `cancel_event.is_set()` every iteration. Send back a Pillow image or NumPy result through the queue, then convert and display it only in the queue-polling method on the Tkinter thread.[cite:23][cite:26][cite:29]

## Debouncing resize

### Goal

A `<Configure>` binding can fire repeatedly while the user drags the window border, so expensive image preview work should be delayed until the event stream quiets down.[cite:24][cite:32] Tkinter's `after()` schedules the deferred callback and `after_cancel()` cancels the previously scheduled one, which makes it a standard debounce mechanism.[cite:24]

### Required skills

- Tkinter event binding and widget lifecycle.
- Deferred callback management with `after()` IDs.
- Lightweight separation between cheap resize bookkeeping and expensive redraw work.[cite:24][cite:32]

### Minimal pattern

```python
import tkinter as tk


class App:
    def __init__(self, root):
        self.root = root
        self.resize_after_id = None
        self.root.bind("<Configure>", self.on_configure)

    def on_configure(self, event):
        if event.widget is not self.root:
            return
        if self.resize_after_id is not None:
            self.root.after_cancel(self.resize_after_id)
        self.resize_after_id = self.root.after(200, self.handle_resize_done)

    def handle_resize_done(self):
        self.resize_after_id = None
        print("Resize settled; do expensive redraw now")


root = tk.Tk()
App(root)
root.mainloop()
```

### How to apply it to the dithering app

The current preview refresh should not run directly inside `_on_resize`. Instead, `_on_resize` should only restart a 150 to 250 ms timer, and the delayed callback should call the thumbnail-generation and label-refresh logic once after the user stops dragging.[cite:24][cite:32]

## Recommended implementation sequence

1. Refactor the dithering code into a pure NumPy processing module using the scanline implementation first, because it reduces the main compute cost and also makes threading easier.[cite:18][cite:22]
2. Add threaded execution with a cancel event and queue-based result delivery, because once the worker is isolated the GUI no longer freezes during long runs.[cite:23][cite:26][cite:29]
3. Debounce resize-triggered preview regeneration, because this removes redundant redraws that otherwise multiply the perceived slowness during window interaction.[cite:24][cite:32]
4. After all three changes, test with large images and repeated resize/open/save cycles to confirm correctness, responsiveness, and clean cancellation behavior.[cite:18][cite:24][cite:26]

## Acceptance checklist

- Dithering output visually matches the old implementation for the same kernel and bin count on a fixed test image.[cite:1][cite:2]
- The GUI remains responsive while processing a large image.[cite:26][cite:29]
- Cancel stops the job within at most a few scanlines rather than waiting for the full image to finish.[cite:23][cite:26]
- Resizing the window does not trigger repeated expensive redraws while dragging.[cite:24][cite:32]
- All widget updates occur on the Tkinter thread only.[cite:26][cite:29]
