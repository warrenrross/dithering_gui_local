import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np

try:
    import numba
    _NUMBA = True

    @numba.njit(cache=True, nogil=True)
    def _process_row_exact(buf, out_row, y, levels, dxdy, weights, h, w):
        """JIT-compiled per-pixel diffusion for one row — exact same-row propagation."""
        for x in range(w):
            old = buf[y, x]
            new_val = np.floor(old * levels + 0.5) / levels
            err = old - new_val
            out_row[x] = new_val
            for k in range(dxdy.shape[0]):
                nx = x + dxdy[k, 0]
                ny = y + dxdy[k, 1]
                if 0 <= nx < w and 0 <= ny < h:
                    buf[ny, nx] += err * weights[k]

except ImportError:
    _NUMBA = False


def diffuse_channel_exact(channel, algo_name, bins, cancel_event=None, progress_fn=None):
    """Error diffusion with exact same-row propagation (Numba) or fast approximation fallback."""
    if not _NUMBA:
        return diffuse_channel_fast(channel, algo_name, bins, cancel_event, progress_fn)

    kernel = ALGORITHMS[algo_name]
    dxdy    = np.array([[dx, dy] for dx, dy, _ in kernel["offsets"]], dtype=np.int32)
    weights = np.array([w / kernel["divisor"] for *_, w in kernel["offsets"]], dtype=np.float32)

    buf = channel.astype(np.float32) / 255.0
    out = np.empty_like(buf)
    h, w = buf.shape
    levels = np.float32(max(2, int(bins)) - 1)

    for y in range(h):
        if cancel_event is not None and cancel_event.is_set():
            return None
        if progress_fn is not None:
            progress_fn(y, h)
        _process_row_exact(buf, out[y], y, levels, dxdy, weights, h, w)

    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _prewarm_numba():
    """Trigger Numba JIT compilation on a tiny array so first real dither is instant."""
    if not _NUMBA:
        return
    dummy = np.full((4, 4), 128, dtype=np.uint8)
    diffuse_channel_exact(dummy, "Floyd-Steinberg", 2)

ALGORITHMS = {
    "Floyd-Steinberg": {
        "divisor": 16,
        "offsets": [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)],
    },
    "Jarvis, Judice, and Ninke": {
        "divisor": 48,
        "offsets": [(1, 0, 7), (2, 0, 5), (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3), (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)],
    },
    "Stucki": {
        "divisor": 42,
        "offsets": [(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2), (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)],
    },
    "Burkes": {
        "divisor": 32,
        "offsets": [(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)],
    },
}

ZOOM_STEPS = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]


def diffuse_channel_fast(channel, algo_name, bins, cancel_event=None, progress_fn=None):
    """Scanline-based NumPy error diffusion. Returns uint8 array, or None if cancelled.

    Uses a separate accumulator (buf) and output (out) so that same-row error
    additions never corrupt already-quantized output pixels. Same-row (dy==0)
    offsets are skipped; inter-row propagation is exact.
    """
    kernel = ALGORITHMS[algo_name]
    offsets = [(dx, dy, w / kernel["divisor"]) for dx, dy, w in kernel["offsets"]]

    buf = channel.astype(np.float32) / 255.0   # accumulates inter-row errors
    out = np.empty_like(buf)                    # receives clean quantized values
    h, w = buf.shape
    levels = max(2, int(bins)) - 1

    for y in range(h):
        if cancel_event is not None and cancel_event.is_set():
            return None
        if progress_fn is not None:
            progress_fn(y, h)

        old_row = buf[y].copy()
        quantized = np.round(old_row * levels) / levels
        err = old_row - quantized
        out[y] = quantized

        for dx, dy, weight in offsets:
            if dy == 0:
                continue          # same-row: skip to avoid corrupting out[y]
            ny = y + dy
            if ny < 0 or ny >= h:
                continue
            if dx == 0:
                buf[ny, :] += err * weight
            elif dx > 0:
                buf[ny, dx:] += err[:-dx] * weight
            else:
                buf[ny, :dx] += err[-dx:] * weight

    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def dither_image(image, grayscale, bins, algo_name, cancel_event=None, progress_fn=None):
    """Dither a PIL Image. Returns PIL Image, or None if cancelled.

    Grayscale: single channel, sequential.
    Color: R/G/B processed in parallel via ThreadPoolExecutor; progress is
    thread-safe via a shared row counter protected by a lock.
    """
    if grayscale:
        channel = np.array(ImageOps.grayscale(image), dtype=np.uint8)
        out = diffuse_channel_exact(channel, algo_name, bins, cancel_event, progress_fn)
        return None if out is None else Image.fromarray(out)

    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    h = arr.shape[0]
    total = 3 * h

    # Thread-safe row counter: each channel increments it once per scanline.
    if progress_fn is not None:
        _lock = threading.Lock()
        _tally = [0]
        def _make_ch_fn():
            def _fn(_y, _h):
                with _lock:
                    _tally[0] += 1
                    progress_fn(_tally[0], total)
            return _fn
    else:
        def _make_ch_fn():
            return None

    results = [None, None, None]

    def _process(i):
        return diffuse_channel_exact(
            arr[:, :, i].copy(), algo_name, bins, cancel_event, _make_ch_fn())

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_process, i) for i in range(3)]
        for i, fut in enumerate(futures):
            result = fut.result()
            if result is None:
                return None          # cancelled; executor waits for others to exit
            results[i] = result

    return Image.fromarray(np.stack(results, axis=-1))


class DitherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Dithering App")
        self.root.geometry("1400x860")
        self.root.minsize(1100, 700)

        self.original_image = None
        self.processed_image = None
        self.original_tk = None
        self.processed_tk = None

        self.mode_var = tk.StringVar(value="Color")
        self.bins_var = tk.IntVar(value=2)
        self.algorithm_var = tk.StringVar(value="Floyd-Steinberg")
        _mode = "exact (Numba)" if _NUMBA else "fast (install numba for exact)"
        self.status_var = tk.StringVar(value=f"Open an image to begin.\nMode: {_mode}")
        self._zoom_factor = None

        self._q = queue.Queue()
        self._cancel_event = None
        self._worker = None
        self._poll_id = None
        self._resize_after_id = None
        self._progress_var = tk.IntVar(value=0)

        self._build_ui()
        self.root.bind("<Configure>", self._on_resize)
        threading.Thread(target=_prewarm_numba, daemon=True).start()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── Controls panel ────────────────────────────────────────────────
        controls = ttk.Frame(self.root, padding=14)
        controls.grid(row=0, column=0, sticky="ns")
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Image Dithering", font=("TkDefaultFont", 14, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 12))

        self._btn_open   = ttk.Button(controls, text="Open Image…",       command=self.open_image)
        self._btn_dither = ttk.Button(controls, text="Dither Image",      command=self.run_dither)
        self._btn_cancel = ttk.Button(controls, text="Cancel",            command=self._cancel_dither, state="disabled")
        self._btn_save   = ttk.Button(controls, text="Save Dithered As…", command=self.save_image)
        self._btn_open.grid(  row=1, column=0, sticky="ew", pady=2)
        self._btn_dither.grid(row=2, column=0, sticky="ew", pady=2)
        self._btn_cancel.grid(row=3, column=0, sticky="ew", pady=2)

        self._progress_bar = ttk.Progressbar(
            controls, variable=self._progress_var, maximum=100, length=200)
        self._progress_bar.grid(row=4, column=0, sticky="ew", pady=(2, 8))

        self._btn_save.grid(row=5, column=0, sticky="ew", pady=2)

        ttk.Separator(controls).grid(row=6, column=0, sticky="ew", pady=12)

        ttk.Label(controls, text="Mode").grid(row=7, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Color",     variable=self.mode_var, value="Color").grid(    row=8,  column=0, sticky="w")
        ttk.Radiobutton(controls, text="Greyscale", variable=self.mode_var, value="Greyscale").grid(row=9,  column=0, sticky="w")

        ttk.Label(controls, text="Color bins / grey levels").grid(row=10, column=0, sticky="w", pady=(12, 0))
        ttk.Spinbox(controls, from_=2, to=64, textvariable=self.bins_var, width=8).grid(
            row=11, column=0, sticky="w", pady=(4, 4))
        ttk.Label(controls, text="Lower = stronger dithering\nHigher = more tones",
                  foreground="#555").grid(row=12, column=0, sticky="w")

        ttk.Label(controls, text="Algorithm").grid(row=13, column=0, sticky="w", pady=(12, 0))
        ttk.Combobox(controls, textvariable=self.algorithm_var, state="readonly",
                     values=list(ALGORITHMS.keys()), width=28).grid(row=14, column=0, sticky="ew", pady=(4, 4))

        ttk.Separator(controls).grid(row=15, column=0, sticky="ew", pady=12)

        ttk.Label(controls, text="Zoom").grid(row=16, column=0, sticky="w")
        zoom_frame = ttk.Frame(controls)
        zoom_frame.grid(row=17, column=0, sticky="w", pady=(4, 0))
        ttk.Button(zoom_frame, text="−", width=3, command=self._zoom_out).pack(side="left")
        self._zoom_entry_var = tk.StringVar(value="Fit")
        zoom_entry = ttk.Entry(zoom_frame, textvariable=self._zoom_entry_var, width=6, justify="center")
        zoom_entry.pack(side="left", padx=4)
        zoom_entry.bind("<Return>",   self._on_zoom_entry)
        zoom_entry.bind("<FocusOut>", self._on_zoom_entry)
        ttk.Button(zoom_frame, text="+", width=3, command=self._zoom_in).pack(side="left")
        ttk.Button(zoom_frame, text="Fit", command=self._zoom_reset).pack(side="left", padx=(8, 0))

        ttk.Separator(controls).grid(row=18, column=0, sticky="ew", pady=12)
        ttk.Label(controls, textvariable=self.status_var, wraplength=240,
                  justify="left").grid(row=19, column=0, sticky="w")

        # ── Preview panel ─────────────────────────────────────────────────
        preview = ttk.Frame(self.root, padding=14)
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)
        preview.rowconfigure(3, weight=1)

        ttk.Label(preview, text="Original", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        self.canvas_orig = tk.Canvas(preview, bg="#2a2a2a", highlightthickness=1,
                                     highlightbackground="#888")
        self.canvas_orig.grid(row=1, column=0, sticky="nsew")

        ttk.Label(preview, text="Dithered", font=("TkDefaultFont", 11, "bold")).grid(
            row=2, column=0, sticky="w", pady=(8, 4))
        self.canvas_proc = tk.Canvas(preview, bg="#2a2a2a", highlightthickness=1,
                                     highlightbackground="#888")
        self.canvas_proc.grid(row=3, column=0, sticky="nsew")

        self.vsb = ttk.Scrollbar(preview, orient="vertical",   command=self._scroll_y)
        self.hsb = ttk.Scrollbar(preview, orient="horizontal", command=self._scroll_x)
        self.vsb.grid(row=1, column=1, rowspan=3, sticky="ns")
        self.hsb.grid(row=4, column=0, sticky="ew")

        self.canvas_orig.configure(
            yscrollcommand=self._yscroll_sync,
            xscrollcommand=self._xscroll_sync,
        )

        for canvas in (self.canvas_orig, self.canvas_proc):
            canvas.bind("<MouseWheel>",       self._on_mousewheel_y)
            canvas.bind("<Shift-MouseWheel>", self._on_mousewheel_x)
            canvas.bind("<Button-4>",         self._on_mousewheel_y)
            canvas.bind("<Button-5>",         self._on_mousewheel_y)

    # ── Scrolling ──────────────────────────────────────────────────────────

    def _scroll_y(self, *args):
        self.canvas_orig.yview(*args)
        self.canvas_proc.yview(*args)

    def _scroll_x(self, *args):
        self.canvas_orig.xview(*args)
        self.canvas_proc.xview(*args)

    def _yscroll_sync(self, lo, hi):
        self.vsb.set(lo, hi)
        self.canvas_proc.yview_moveto(lo)

    def _xscroll_sync(self, lo, hi):
        self.hsb.set(lo, hi)
        self.canvas_proc.xview_moveto(lo)

    def _on_mousewheel_y(self, event):
        if event.num in (4, 5):
            delta = 1 if event.num == 5 else -1
        else:
            delta = int(-event.delta / 120)
        self.canvas_orig.yview_scroll(delta, "units")
        self.canvas_proc.yview_scroll(delta, "units")

    def _on_mousewheel_x(self, event):
        if event.num in (4, 5):
            delta = 1 if event.num == 5 else -1
        else:
            delta = int(-event.delta / 120)
        self.canvas_orig.xview_scroll(delta, "units")
        self.canvas_proc.xview_scroll(delta, "units")

    # ── Zoom ───────────────────────────────────────────────────────────────

    def _nearest_step(self, factor):
        return min(ZOOM_STEPS, key=lambda s: abs(s - factor))

    def _zoom_in(self):
        cur = self._zoom_factor or 1.0
        idx = ZOOM_STEPS.index(self._nearest_step(cur))
        self._zoom_factor = ZOOM_STEPS[min(idx + 1, len(ZOOM_STEPS) - 1)]
        self._apply_zoom()

    def _zoom_out(self):
        cur = self._zoom_factor or 1.0
        idx = ZOOM_STEPS.index(self._nearest_step(cur))
        self._zoom_factor = ZOOM_STEPS[max(idx - 1, 0)]
        self._apply_zoom()

    def _zoom_reset(self):
        self._zoom_factor = None
        self._apply_zoom()

    def _on_zoom_entry(self, event=None):
        raw = self._zoom_entry_var.get().strip().rstrip("%")
        if raw.lower() == "fit":
            self._zoom_factor = None
        else:
            try:
                pct = float(raw)
                self._zoom_factor = max(0.01, min(pct / 100.0, 32.0))
            except ValueError:
                pass
        self._apply_zoom()

    def _apply_zoom(self):
        if self._zoom_factor is None:
            self._zoom_entry_var.set("Fit")
        else:
            self._zoom_entry_var.set(f"{int(self._zoom_factor * 100)}%")
        self._refresh_panels()

    # ── Image operations ───────────────────────────────────────────────────

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
            self.processed_image = None
            self.processed_tk = None
            self.canvas_proc.delete("all")
            self._refresh_panels()
            self.status_var.set(
                f"Loaded: {os.path.basename(path)}\nSelect options and click Dither Image.")
        except Exception as exc:
            messagebox.showerror("Open image", f"Could not open image:\n{exc}")

    def run_dither(self):
        if self.original_image is None:
            return
        if self._worker is not None and self._worker.is_alive():
            return

        bins = max(2, min(64, int(self.bins_var.get())))
        self.bins_var.set(bins)
        grayscale = self.mode_var.get() == "Greyscale"
        algo = self.algorithm_var.get()

        self._cancel_event = threading.Event()
        self._progress_var.set(0)
        self._btn_dither.config(state="disabled")
        self._btn_open.config(state="disabled")
        self._btn_cancel.config(state="normal")
        _mode = "exact" if _NUMBA else "fast"
        self.status_var.set(f"Dithering ({_mode})…")

        self._worker = threading.Thread(
            target=self._dither_worker,
            args=(self.original_image.copy(), grayscale, bins, algo),
            daemon=True,
        )
        self._worker.start()
        self._poll_queue()

    def _dither_worker(self, image, grayscale, bins, algo_name):
        try:
            last_pct = [-1]

            def progress_fn(done, total):
                pct = int(done * 100 / total) if total > 0 else 0
                if pct > last_pct[0]:
                    last_pct[0] = pct
                    self._q.put(("progress", pct))

            result = dither_image(image, grayscale, bins, algo_name, self._cancel_event, progress_fn)
            if result is None:
                self._q.put(("cancelled", None))
            else:
                self._q.put(("done", result))
        except Exception as exc:
            self._q.put(("error", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "progress":
                    self._progress_var.set(payload)
                elif kind == "done":
                    self.processed_image = payload
                    self._refresh_panels()
                    mode = "greyscale" if self.mode_var.get() == "Greyscale" else "color"
                    self.status_var.set(
                        f"Done: {self.algorithm_var.get()} | {mode} | {self.bins_var.get()} bins")
                    self._progress_var.set(100)
                    self._reset_controls()
                    return
                elif kind == "cancelled":
                    self.status_var.set("Cancelled.")
                    self._progress_var.set(0)
                    self._reset_controls()
                    return
                elif kind == "error":
                    messagebox.showerror("Dither image", f"Could not process image:\n{payload}")
                    self.status_var.set("Error during dithering.")
                    self._progress_var.set(0)
                    self._reset_controls()
                    return
        except queue.Empty:
            pass

        if self._worker and self._worker.is_alive():
            self._poll_id = self.root.after(50, self._poll_queue)
        else:
            self._poll_id = None
            self._reset_controls()

    def _reset_controls(self):
        self._btn_dither.config(state="normal")
        self._btn_open.config(state="normal")
        self._btn_cancel.config(state="disabled")

    def _cancel_dither(self):
        if self._cancel_event is not None:
            self._cancel_event.set()

    def save_image(self):
        if self.processed_image is None:
            messagebox.showinfo("Save image", "Open and dither an image first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save dithered image as",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg;*.jpeg")],
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            img = self.processed_image
            if ext in (".jpg", ".jpeg") and img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            img.save(path, quality=95) if ext in (".jpg", ".jpeg") else img.save(path)
            self.status_var.set(f"Saved: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Save image", f"Could not save image:\n{exc}")

    # ── Display ────────────────────────────────────────────────────────────

    def _on_resize(self, event=None):
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(200, self._handle_resize_done)

    def _handle_resize_done(self):
        self._resize_after_id = None
        if self.original_image is not None:
            self._refresh_panels()

    def _refresh_panels(self):
        if self.original_image is None:
            return

        if self._zoom_factor is None:
            self.canvas_orig.update_idletasks()
            self.canvas_proc.update_idletasks()
            cw = max(min(self.canvas_orig.winfo_width(),  self.canvas_proc.winfo_width()),  100)
            ch = max(min(self.canvas_orig.winfo_height(), self.canvas_proc.winfo_height()), 100)

            ow, oh = self.original_image.size
            scale = min(cw / ow, ch / oh)
            tw, th = max(1, int(ow * scale)), max(1, int(oh * scale))

            orig = self.original_image.copy()
            orig.thumbnail((tw, th), Image.Resampling.LANCZOS)
            ox = (cw - orig.width) // 2
            oy = (ch - orig.height) // 2
            self._draw(self.canvas_orig, orig, "original_tk", ox, oy, cw, ch)

            if self.processed_image is not None:
                proc = self.processed_image.copy()
                proc.thumbnail((tw, th), Image.Resampling.LANCZOS)
                self._draw(self.canvas_proc, proc, "processed_tk", ox, oy, cw, ch)
        else:
            factor = self._zoom_factor
            ow, oh = self.original_image.size
            tw, th = max(1, int(ow * factor)), max(1, int(oh * factor))

            orig = self.original_image.resize((tw, th), Image.Resampling.LANCZOS)
            self._draw(self.canvas_orig, orig, "original_tk", 0, 0, tw, th)

            if self.processed_image is not None:
                proc = self.processed_image.resize((tw, th), Image.Resampling.LANCZOS)
                self._draw(self.canvas_proc, proc, "processed_tk", 0, 0, tw, th)

    def _draw(self, canvas, image, attr, x, y, region_w, region_h):
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        photo = ImageTk.PhotoImage(image)
        setattr(self, attr, photo)
        canvas.delete("all")
        canvas.create_image(x, y, anchor="nw", image=photo)
        canvas.configure(scrollregion=(0, 0, region_w, region_h))


if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    app = DitherApp(root)
    root.mainloop()
