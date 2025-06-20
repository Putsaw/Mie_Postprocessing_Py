import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk
import numpy as np
from zoom_utils import enlarge_image
from cine_utils import CineReader
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class VideoAnnotatorUI:
    def __init__(self, master):
        self.master = master
        master.title("Cine Video Annotator")

        # Layout parameters
        self.layout_positions = {
            'load_btn': 0,
            'frame_label': 1,
            'prev_btn': 2,
            'next_btn': 3,
            'param_start_col': 4,
            'confirm_btn': 12
        }

        # Video reader and annotation data
        self.reader = CineReader()
        self.total_frames = 0
        self.current_index = 0
        # Integer zoom factor for pixel-perfect scaling
        self.zoom_factor = 1
        self.orig_img = np.zeros_like       # PIL Image after processing
        self.mask = np.zeros_like           # Full-frame mask (H×W)

        # Paint settings
        self.brush_color = (255, 0, 0)  # default red
        # Alpha controlled by GUI
        self.alpha_var = tk.IntVar(value=50)

        # Brush controls
        self.brush_shape = tk.StringVar(value='circle')
        self.brush_size = tk.IntVar(value=20)

        # Build interface
        self._build_controls(master)
        self._build_content(master)

    def _build_controls(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lp = self.layout_positions

        # Load / navigation
        self.load_btn = ttk.Button(ctrl, text="Load Video", command=self.load_video)
        self.load_btn.grid(row=0, column=lp['load_btn'], padx=2)
        self.frame_label = ttk.Label(ctrl, text="Frame: 0/0")
        self.frame_label.grid(row=0, column=lp['frame_label'], padx=5)
        self.frame_entry = tk.IntVar(value=1)
        ttk.Entry(ctrl, textvariable=self.frame_entry, width=5).grid(row=0, column=lp['frame_label']+1)
        ttk.Button(ctrl, text="Go", command=self._on_go_frame).grid(row=0, column=lp['frame_label']+2)

        self.prev_btn = ttk.Button(ctrl, text="Prev Frame", command=self.prev_frame, state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=lp['prev_btn'], padx=2)
        self.next_btn = ttk.Button(ctrl, text="Next Frame", command=self.next_frame, state=tk.DISABLED)
        self.next_btn.grid(row=0, column=lp['next_btn'], padx=2)

        # Display params
        self.vars = {}
        for i,name in enumerate(["Gain","Gamma","Black","White"]):
            ttk.Label(ctrl, text=f"{name}:").grid(row=0, column=lp['param_start_col']+i*2)
            v = tk.DoubleVar(value=1.0 if name in ("Gain","Gamma") else 0.0)
            ttk.Entry(ctrl, textvariable=v, width=5).grid(row=0, column=lp['param_start_col']+i*2+1)
            self.vars[name.lower()] = v

        # Apply
        self.confirm_btn = ttk.Button(ctrl, text="Apply", command=self.update_image, state=tk.DISABLED)
        self.confirm_btn.grid(row=0, column=lp['confirm_btn'], padx=5)

        # Paint controls on second row
        bc = lp['param_start_col']
        ttk.Label(ctrl, text="Brush:").grid(row=1, column=bc, pady=(5,0))
        ttk.Combobox(ctrl, textvariable=self.brush_shape, values=['circle','square'], width=10).grid(row=1, column=bc+1, pady=(5,0))
        ttk.Label(ctrl, text="Size:").grid(row=1, column=bc+4, pady=(5,0))
        ttk.Entry(ctrl, textvariable=self.brush_size, width=5).grid(row=1, column=bc+5, pady=(5,0))
        # Color chooser
        ttk.Button(ctrl, text="Color", command=self.choose_color).grid(row=1, column=bc+7, padx=(10,0), pady=(5,0))
        # Alpha slider
        ttk.Label(ctrl, text="Alpha:").grid(row=1, column=bc+8, pady=(5,0))
        tk.Scale(ctrl, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.alpha_var, length=100).grid(
            row=1, column=bc+9, pady=(5,0), sticky='w')

    def choose_color(self):
        color = colorchooser.askcolor(color='#%02x%02x%02x' % self.brush_color, title='Select brush color')
        if color[0]:
            self.brush_color = tuple(map(int, color[0]))

    def _build_content(self, parent):
        content = ttk.Frame(parent)
        content.pack(fill=tk.BOTH, expand=True)
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)

        # Canvas + scrollbars
        cf = ttk.Frame(content)
        cf.grid(row=0, column=0, sticky='nsew')
        self.canvas = tk.Canvas(cf, bg='black')
        self.hbar = ttk.Scrollbar(cf, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(cf, orient=tk.VERTICAL,   command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.hbar.grid(row=1, column=0, sticky='we')
        self.vbar.grid(row=0, column=1, sticky='ns')
        cf.rowconfigure(0, weight=1); cf.columnconfigure(0, weight=1)

        # Histogram
        hf = ttk.Frame(content)
        hf.grid(row=0, column=1, sticky='ns', padx=5, pady=5)
        hf.rowconfigure(0, weight=1); hf.columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(3,3)); self.ax = self.fig.add_subplot(111)
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=hf)
        self.canvas_hist.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Bindings
        self.canvas.bind('<MouseWheel>',    self._on_zoom)
        self.canvas.bind('<Button-4>',      self._on_zoom)
        self.canvas.bind('<Button-5>',      self._on_zoom)
        self.canvas.bind('<B1-Motion>',     lambda e: self._on_paint(e, True))
        self.canvas.bind('<ButtonPress-1>', lambda e: self._on_paint(e, True))
        self.canvas.bind('<B3-Motion>',     lambda e: self._on_paint(e, False))
        self.canvas.bind('<ButtonPress-3>', lambda e: self._on_paint(e, False))

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[('Cine','*.cine')])
        try:
            self.reader.load(path)
        except Exception as e:
            messagebox.showerror('Error', f'Cannot load video:\n{e}')
            return

        self.total_frames = self.reader.frame_count
        self.mask = np.zeros((self.reader.height, self.reader.width), dtype=np.uint8)
        self.current_index = 0
        self.mask = np.zeros((self.reader.height, self.reader.width), dtype=np.uint8)
        for w in (self.prev_btn, self.next_btn, self.confirm_btn): w.config(state=tk.NORMAL)
        
        self.update_image()

    def prev_frame(self):
        if self.current_index>0:
            self.current_index-=1; self.mask = np.zeros_like(self.mask); self.update_image()

    def next_frame(self):
        if self.current_index<self.total_frames-1:
            self.current_index+=1; self.mask = np.zeros_like(self.mask); self.update_image()

    def update_image(self):
        frame = self.reader.read_frame(self.current_index).astype(np.float32)
        if self.mask is None or self.mask.shape != frame.shape:
            self.mask = np.zeros(frame.shape, dtype=np.uint8)
        g, gm, bl, wh = [self.vars[k].get() for k in ('gain','gamma','black','white')]
        img = frame/4096 * g
        img = np.clip(img,0,1)
        if gm>0 and gm!=1: img=img**gm
        img8 = (img*255).astype(np.uint8)
        if wh>bl and bl>=0:
            img8 = np.clip((img8-bl)*(255/(wh-bl)),0,255).astype(np.uint8)
        self.orig_img = Image.fromarray(img8)
        self.ax.clear(); self.ax.hist(img8.ravel(),bins=256); self.ax.set_title('Processed Histogram'); self.canvas_hist.draw()
        self._draw_scaled(); self.frame_label.config(text=f"Frame: {self.current_index+1}/{self.total_frames}")

    def _draw_scaled(self):
        base = self.orig_img.convert('RGBA')
        overlay = Image.new('RGBA', base.size, (*self.brush_color, self.alpha_var.get()))
        mask_img = Image.fromarray((self.mask * 255).astype(np.uint8)).convert('L')

        composited = base.copy()
        composited.paste(overlay, (0, 0), mask_img)

        scaled = enlarge_image(composited, int(self.zoom_factor))
        w2, h2 = scaled.size

        self.photo = ImageTk.PhotoImage(scaled)
        self.canvas.delete('IMG')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo, tags='IMG')
        self.canvas.config(scrollregion=(0, 0, w2, h2))

    def _on_zoom(self, event):
        """Zoom in or out in integer steps using the mouse wheel."""
        direction = 1 if getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4 else -1
        self.zoom_factor = max(1, self.zoom_factor + direction)
        self._draw_scaled()

    def _on_paint(self,event,paint=True):
        x=int(self.canvas.canvasx(event.x)/self.zoom_factor); y=int(self.canvas.canvasy(event.y)/self.zoom_factor)
        size=self.brush_size.get()
        if self.brush_shape.get()=='circle':
            yy,xx=np.ogrid[-y:self.mask.shape[0]-y, -x:self.mask.shape[1]-x]; mask_area = xx*xx+yy*yy<=size*size
            self.mask[mask_area] = 1 if paint else 0
        else:
            x0,x1=max(0,x-size),min(self.mask.shape[1],x+size)
            y0,y1=max(0,y-size),min(self.mask.shape[0],y+size)
            self.mask[y0:y1,x0:x1] = 1 if paint else 0
        self._draw_scaled()
    
    def _on_go_frame(self):
        """Jump to a specified frame number from entry box."""
        idx = self.frame_entry.get() - 1
        if 0 <= idx < self.total_frames:
            self.current_index = idx
            self.mask = np.zeros_like(self.mask)
            self.update_image()

if __name__=='__main__':
    root=tk.Tk(); app=VideoAnnotatorUI(root); root.mainloop()


