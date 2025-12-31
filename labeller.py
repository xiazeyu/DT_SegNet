# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "numpy>=2.3.5",
#     "pillow>=12.0.0",
# ]
# ///
import tkinter as tk
from tkinter import messagebox, Toplevel, ttk
from PIL import Image, ImageTk, ImageEnhance
import os
import glob
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import platform

# Configuration
INPUT_FOLDER = "input"
LABEL_EXTENSION = ".txt"

class PrecipitateLabeller:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1400x900")
        self.update_title()

        # Data State
        self.image_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.png")))
        self.current_file_path = None
        self.labels = []  # List of tuples: [(x1, y1, x2, y2), ...] normalized
        self.current_image = None
        self.display_image = None
        self.tk_image = None
        
        # Drawing State
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.temp_line_id = None
        
        # Image Adjustments
        self.brightness = 1.0
        self.contrast = 1.0
        self.adj_window = None

        if not self.image_files:
            messagebox.showerror("Error", f"No .png images found in '{INPUT_FOLDER}' folder.")
            root.destroy()
            return

        self._setup_ui()
        self.refresh_file_list()
        
        # Select first image by default
        if self.image_files:
            self.listbox.selection_set(0)
            self.load_image_from_list(None)

    def update_title(self):
        self.root.title("SEM Labeller | Keys: [Up/Down] Next File | [Right-Click] Undo | [A] Adjust")

    def _setup_ui(self):
        # --- Layout Containers ---
        # Left Panel: File List
        left_panel = tk.Frame(self.root, width=250, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right Panel: Canvas & Toolbar
        right_panel = tk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Sidebar (File List) ---
        tk.Label(left_panel, text="Image List", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=5)
        
        scrollbar_lb = tk.Scrollbar(left_panel)
        scrollbar_lb.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(left_panel, selectmode=tk.SINGLE, yscrollcommand=scrollbar_lb.set, font=("Arial", 10))
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_lb.config(command=self.listbox.yview)
        
        self.listbox.bind("<<ListboxSelect>>", self.load_image_from_list)

        # --- Top Toolbar ---
        toolbar = tk.Frame(right_panel, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.lbl_filename = tk.Label(toolbar, text="No Image Selected", font=("Arial", 12, "bold"))
        self.lbl_filename.pack(side=tk.LEFT, padx=10, pady=5)

        tk.Button(toolbar, text="Run Analysis", command=self.run_analysis, bg="lightblue").pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Button(toolbar, text="Adjust Image (A)", command=self.toggle_adj_window).pack(side=tk.RIGHT, padx=5)

        # --- Main Canvas with Scrollbars ---
        # Frame to hold canvas and scrollbars
        canvas_frame = tk.Frame(right_panel, bg="gray")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        self.v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)

        # Canvas
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="cross",
                                xscrollcommand=self.h_scroll.set,
                                yscrollcommand=self.v_scroll.set)

        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Bindings ---
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Right Click Undo (Platform safe)
        if platform.system() == "Darwin": # macOS
            self.canvas.bind("<Button-2>", self.undo_last_label)
        else: # Windows / Linux
            self.canvas.bind("<Button-3>", self.undo_last_label)

        # Keyboard Shortcuts
        self.root.bind("a", lambda event: self.toggle_adj_window())
        self.root.bind("<Up>", self.on_arrow_up)
        self.root.bind("<Down>", self.on_arrow_down)

    def refresh_file_list(self):
        self.listbox.delete(0, tk.END)
        for idx, filepath in enumerate(self.image_files):
            filename = os.path.basename(filepath)
            self.listbox.insert(tk.END, filename)
            
            # Check if labeled
            txt_path = self.get_label_path(filepath)
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                self.listbox.itemconfig(idx, {'fg': 'green'})
            else:
                self.listbox.itemconfig(idx, {'fg': 'black'})

    # --- Navigation ---
    def on_arrow_up(self, event):
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0]
            if idx > 0:
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(idx - 1)
                self.listbox.see(idx - 1)
                self.load_image_from_list(None)

    def on_arrow_down(self, event):
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0]
            if idx < len(self.image_files) - 1:
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(idx + 1)
                self.listbox.see(idx + 1)
                self.load_image_from_list(None)
        elif self.image_files:
            self.listbox.selection_set(0)
            self.load_image_from_list(None)

    # --- Image Loading ---
    def load_image_from_list(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        filepath = self.image_files[index]
        
        self.current_file_path = filepath
        self.lbl_filename.config(text=f"{os.path.basename(filepath)}")
        
        # Load Image
        self.current_image = Image.open(filepath).convert("RGB")
        self.img_w, self.img_h = self.current_image.size
        
        # Configure scrollregion for new image size
        self.canvas.config(scrollregion=(0, 0, self.img_w, self.img_h))
        
        # Load Labels
        self.load_labels(filepath)
        
        # Update Display
        self.update_image_appearance()

    def toggle_adj_window(self):
        if self.adj_window is not None and tk.Toplevel.winfo_exists(self.adj_window):
            self.adj_window.destroy()
            self.adj_window = None
            return

        self.adj_window = Toplevel(self.root)
        self.adj_window.title("Adjust")
        self.adj_window.geometry("250x150")
        self.adj_window.resizable(False, False)
        self.adj_window.attributes("-topmost", True)

        tk.Label(self.adj_window, text="Brightness").pack(pady=(10, 0))
        slider_bright = tk.Scale(self.adj_window, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.set_brightness)
        slider_bright.set(self.brightness)
        slider_bright.pack(fill=tk.X, padx=10)

        tk.Label(self.adj_window, text="Contrast").pack(pady=(5, 0))
        slider_cont = tk.Scale(self.adj_window, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.set_contrast)
        slider_cont.set(self.contrast)
        slider_cont.pack(fill=tk.X, padx=10)

    def set_brightness(self, val):
        self.brightness = float(val)
        self.update_image_appearance()

    def set_contrast(self, val):
        self.contrast = float(val)
        self.update_image_appearance()

    def update_image_appearance(self, _=None):
        if not self.current_image:
            return
            
        enhancer = ImageEnhance.Brightness(self.current_image)
        img = enhancer.enhance(self.brightness)
        enhancer = ImageEnhance.Contrast(img)
        self.display_image = enhancer.enhance(self.contrast)
        
        self.redraw()

    def redraw(self):
        if not self.display_image:
            return

        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        
        # Draw image at 0,0
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        
        # Draw Labels
        for (x1, y1, x2, y2) in self.labels:
            sx1, sy1 = x1 * self.img_w, y1 * self.img_h
            sx2, sy2 = x2 * self.img_w, y2 * self.img_h
            
            if abs(sx1 - sx2) < 2 and abs(sy1 - sy2) < 2:
                r = 3
                self.canvas.create_oval(sx1-r, sy1-r, sx1+r, sy1+r, fill="yellow", outline="yellow")
            else:
                self.canvas.create_line(sx1, sy1, sx2, sy2, fill="red", width=2)

    # --- Mouse Interaction ---
    def on_mouse_down(self, event):
        # Translate window coord to canvas coord
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        if 0 <= cx <= self.img_w and 0 <= cy <= self.img_h:
            self.drawing = True
            self.start_x = cx / self.img_w
            self.start_y = cy / self.img_h

    def on_mouse_drag(self, event):
        if self.drawing:
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            
            if self.temp_line_id:
                self.canvas.delete(self.temp_line_id)
            
            sx1 = self.start_x * self.img_w
            sy1 = self.start_y * self.img_h
            
            self.temp_line_id = self.canvas.create_line(sx1, sy1, cx, cy, fill="green", width=2)

    def on_mouse_up(self, event):
        if self.drawing:
            self.drawing = False
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            
            if self.temp_line_id:
                self.canvas.delete(self.temp_line_id)
                self.temp_line_id = None
            
            cx = max(0, min(cx, self.img_w))
            cy = max(0, min(cy, self.img_h))
            
            end_x = cx / self.img_w
            end_y = cy / self.img_h
            
            self.labels.append((self.start_x, self.start_y, end_x, end_y))
            self.save_labels()
            self.redraw()
            
            # Update list color
            if self.listbox.curselection():
                self.listbox.itemconfig(self.listbox.curselection()[0], {'fg': 'green'})

    def undo_last_label(self, event):
        if self.labels:
            self.labels.pop()
            self.save_labels()
            self.redraw()
            if not self.labels and self.listbox.curselection():
                self.listbox.itemconfig(self.listbox.curselection()[0], {'fg': 'black'})

    # --- I/O ---
    def get_label_path(self, img_path):
        base, _ = os.path.splitext(img_path)
        return base + LABEL_EXTENSION

    def save_labels(self):
        if not self.current_file_path: return
        txt_path = self.get_label_path(self.current_file_path)
        try:
            with open(txt_path, 'w') as f:
                for l in self.labels:
                    f.write(f"{l[0]} {l[1]} {l[2]} {l[3]}\n")
        except Exception as e:
            print(f"Error saving: {e}")

    def load_labels(self, img_path):
        self.labels = []
        txt_path = self.get_label_path(img_path)
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        if len(parts) == 4:
                            self.labels.append(tuple(parts))
            except Exception as e:
                print(f"Error loading labels: {e}")

    # --- Analysis ---
    def run_analysis(self):
        txt_files = glob.glob(os.path.join(INPUT_FOLDER, "*" + LABEL_EXTENSION))
        if not txt_files:
            messagebox.showinfo("Analysis", "No label files found.")
            return

        self.analysis_data = {} 

        for txt_file in txt_files:
            img_path = txt_file.replace(LABEL_EXTENSION, ".png")
            filename = os.path.basename(img_path)
            
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    w, h = img.size
            else:
                w, h = 1000, 1000

            diameters = []
            with open(txt_file, 'r') as f:
                for line in f:
                    x1, y1, x2, y2 = map(float, line.strip().split())
                    px1, py1 = x1 * w, y1 * h
                    px2, py2 = x2 * w, y2 * h
                    d = math.sqrt((px2 - px1)**2 + (py2 - py1)**2)
                    diameters.append(d)
            
            avg_d = np.mean(diameters) if diameters else 0
            self.analysis_data[filename] = {
                'count': len(diameters),
                'avg': avg_d,
                'diameters': diameters
            }

        # Analysis Window
        self.an_window = Toplevel(self.root)
        self.an_window.title("Analysis Dashboard")
        self.an_window.geometry("900x600")

        # Layout
        left_frame = tk.Frame(self.an_window, width=300, bg="white")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        right_frame = tk.Frame(self.an_window, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(left_frame, text="Select Image", font=("bold")).pack(pady=5)
        
        self.an_listbox = tk.Listbox(left_frame)
        self.an_listbox.pack(fill=tk.BOTH, expand=True)
        self.an_listbox.bind("<<ListboxSelect>>", self.update_analysis_plot)

        # Fill listbox
        for fname in sorted(self.analysis_data.keys()):
            self.an_listbox.insert(tk.END, fname)

        # Info & Plot Area
        self.stats_frame = tk.Frame(right_frame, bg="white", bd=2, relief=tk.GROOVE)
        self.stats_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.lbl_stats_count = tk.Label(self.stats_frame, text="Count: --", font=("Arial", 14), bg="white", fg="blue")
        self.lbl_stats_count.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.lbl_stats_avg = tk.Label(self.stats_frame, text="Avg Size: -- px", font=("Arial", 14), bg="white", fg="green")
        self.lbl_stats_avg.pack(side=tk.LEFT, padx=20, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_analysis_plot(self, event):
        selection = self.an_listbox.curselection()
        if not selection: return

        sorted_keys = sorted(self.analysis_data.keys())
        fname = sorted_keys[selection[0]]
        data = self.analysis_data[fname]
        
        # Update Text Stats
        self.lbl_stats_count.config(text=f"Count: {data['count']}")
        self.lbl_stats_avg.config(text=f"Avg Size: {data['avg']:.2f} px")

        # Update Plot
        self.ax.clear()
        if data['diameters']:
            self.ax.hist(data['diameters'], bins=15, color='orange', edgecolor='black', alpha=0.7)
            self.ax.set_title(f"Size Distribution: {fname}")
            self.ax.set_xlabel("Diameter (px)")
            self.ax.set_ylabel("Frequency")
        else:
            self.ax.text(0.5, 0.5, "No labels", ha='center', va='center')
        
        self.canvas_plot.draw()

if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
    
    root = tk.Tk()
    app = PrecipitateLabeller(root)
    root.mainloop()