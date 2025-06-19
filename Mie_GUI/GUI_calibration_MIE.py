"""Small utility to browse frames within a Phantom ``.cine`` video."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
# from cine_utils import CineReader
from cine_utils import *

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

        # Video reader and current frame index
        self.reader = CineReader()
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
            self.reader.load(path)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load video:\n{e}')
            return

        self.current_index = 0

        self.prev_btn.config(state=tk.NORMAL)
        self.next_btn.config(state=tk.NORMAL)

        # Display the very first frame
        self.show_frame(0)

    def read_current(self):
        """Read the current frame from disk and return a ``PIL.Image``."""

        frame = self.reader.read_frame(self.current_index)


        # Convert to 8‑bit for display.  The ``/16`` maps the typical 12‑bit
        # sensor values into 0‑255 range.
        img8 = np.clip(frame / 16, 0, 255).astype(np.uint8)
        return Image.fromarray(img8)

    def show_frame(self, idx):
        """Display the frame at ``idx`` if it exists."""

        if 0 <= idx < self.reader.frame_count:
            self.current_index = idx

            img = self.read_current()
            self.photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.photo)

            # Update status and entry field
            self.info.config(text=f"Frame {idx + 1}/{self.reader.frame_count}")
            self.frame_var.set(idx + 1)

    def prev_frame(self):
        """Show the previous frame if possible."""

        if self.current_index > 0:
            self.show_frame(self.current_index - 1)

    def next_frame(self):
        """Show the next frame if possible."""

        if self.current_index < self.reader.frame_count - 1:
            self.show_frame(self.current_index + 1)

    def goto_frame(self):
        """Jump to the frame specified in the entry box."""

        idx = self.frame_var.get() - 1
        if 0 <= idx < self.reader.frame_count:
            self.show_frame(idx)
        else:
            messagebox.showwarning('Out of range', 'Frame number out of range')

if __name__ == '__main__':
    # When executed directly, open the viewer window.
    root = tk.Tk()
    app = FrameViewer(root)
    root.mainloop()