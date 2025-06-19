import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pycine.file as cine

class FrameViewer:
    """Simple Tkinter widget to preview frames from a ``.cine`` file.

    Only a single frame is read from disk at a time so that large files
    do not need to be fully loaded into memory.  Frames can be navigated
    using "Prev"/"Next" buttons or by entering a specific frame number.
    """
    def __init__(self, master):
        # Keep a reference to the root window so it doesn't get garbage-
        # collected.
        self.master = master
        master.title("Cine Frame Viewer")

        # Video properties
        self.path = None
        self.frame_offsets = []
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.current_index = 0

        self._build_ui(master)

    def _build_ui(self, parent):
        """Create and lay out all widgets."""

        # Container for buttons and entry field
        ctrl = ttk.Frame(parent)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Button used to browse for a ``.cine`` file
        self.load_btn = ttk.Button(ctrl, text="Load Video",
                                   command=self.load_video)
        self.load_btn.grid(row=0, column=0, padx=2)

        # Navigation buttons.  Disabled until a video is loaded
        self.prev_btn = ttk.Button(ctrl, text="Prev",
                                   command=self.prev_frame,
                                   state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=1, padx=2)
        self.next_btn = ttk.Button(ctrl, text="Next",
                                   command=self.next_frame,
                                   state=tk.DISABLED)
        self.next_btn.grid(row=0, column=2, padx=2)

        # Entry widget to jump to a specific frame
        ttk.Label(ctrl, text="Frame:").grid(row=0, column=3, padx=(10, 0))
        self.frame_var = tk.IntVar(value=1)
        self.entry = ttk.Entry(ctrl, textvariable=self.frame_var, width=6)
        self.entry.grid(row=0, column=4)
        ttk.Button(ctrl, text="Go",
                   command=self.goto_frame).grid(row=0, column=5, padx=2)

        # Status text showing the current frame number
        self.info = ttk.Label(ctrl, text="No video loaded")
        self.info.grid(row=0, column=6, padx=(10, 0))

        # Where the image is displayed
        self.img_label = ttk.Label(parent)
        self.img_label.pack(expand=True)


    def load_video(self):
        """Open a ``.cine`` file and initialise video properties."""

        path = filedialog.askopenfilename(filetypes=[('Cine', '*.cine')])
        if not path:
            return

        try:
            header = cine.read_header(path)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load video:\n{e}')
            return

        # Cache properties for quick access while browsing
        self.path = path
        self.frame_offsets = header['pImage']
        self.frame_count = len(self.frame_offsets)
        self.width = header['bitmapinfoheader'].biWidth
        self.height = header['bitmapinfoheader'].biHeight
        self.current_index = 0

        self.prev_btn.config(state=tk.NORMAL)
        self.next_btn.config(state=tk.NORMAL)

        # Display the very first frame
        self.show_frame(0)

    def read_current(self):
        """Read the current frame from disk and return a ``PIL.Image``."""

        offset = self.frame_offsets[self.current_index]

        # ``pycine`` gives the offset to a small per-frame header.  Skip
        # those eight bytes so that we start at the pixel data.
        with open(self.path, "rb") as f:
            f.seek(offset)
            f.read(8)
            data = np.fromfile(f, dtype=np.uint16,
                               count=self.width * self.height)

        # Reshape the 1‑D array to 2‑D and flip vertically because Phantom
        # stores rows bottom‑up.
        frame = data.reshape(self.height, self.width)
        frame = np.flipud(frame)

        # Convert to 8‑bit for display.  The ``/16`` maps the typical 12‑bit
        # sensor values into 0‑255 range.
        img8 = np.clip(frame / 16, 0, 255).astype(np.uint8)
        return Image.fromarray(img8)

    def show_frame(self, idx):
        """Display the frame at ``idx`` if it exists."""

        if 0 <= idx < self.frame_count:
            self.current_index = idx

            img = self.read_current()
            self.photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.photo)

            # Update status and entry field
            self.info.config(text=f"Frame {idx + 1}/{self.frame_count}")
            self.frame_var.set(idx + 1)


    def prev_frame(self):
        """Show the previous frame if possible."""

        if self.current_index > 0:
            self.show_frame(self.current_index - 1)

    def next_frame(self):
        """Show the next frame if possible."""

        if self.current_index < self.frame_count - 1:
            self.show_frame(self.current_index + 1)

    def goto_frame(self):
        """Jump to the frame specified in the entry box."""

        idx = self.frame_var.get() - 1
        if 0 <= idx < self.frame_count:
            self.show_frame(idx)
        else:
            messagebox.showwarning('Out of range', 'Frame number out of range')

if __name__ == '__main__':
    # When executed directly, open the viewer window.
    root = tk.Tk()
    app = FrameViewer(root)
    root.mainloop()