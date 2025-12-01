import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import sys
import threading
from inpainting import InpaintNNF

class InpaintingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("PatchMatch Inpainting GUI")

        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.mask = None
        self.inpainted_result = None
        self.is_showing_result = False
        self.last_x, self.last_y = None, None

        control_frame = tk.Frame(master, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        tk.Label(control_frame, text="File Operations", font=("Arial", 12, "bold")).pack(anchor="w")
        self.btn_load = tk.Button(control_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(fill=tk.X, pady=5)
        self.btn_save = tk.Button(control_frame, text="Save Result", command=self.save_result, state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X)

        tk.Label(control_frame, text="Mask Drawing", font=("Arial", 12, "bold")).pack(anchor="w", pady=(20, 0))
        self.brush_slider = tk.Scale(control_frame, from_=1, to=50, orient=tk.HORIZONTAL, label="Brush Size")
        self.brush_slider.set(10)
        self.brush_slider.pack(fill=tk.X)
        self.erase_mode = tk.BooleanVar()
        self.chk_erase = tk.Checkbutton(control_frame, text="Erase Mask", var=self.erase_mode)
        self.chk_erase.pack(anchor="w")

        tk.Label(control_frame, text="Inpainting Parameters", font=("Arial", 12, "bold")).pack(anchor="w", pady=(20, 0))
        self.patch_size_slider = tk.Scale(control_frame, from_=3, to=15, orient=tk.HORIZONTAL, label="Patch Size", resolution=2)
        self.patch_size_slider.set(7)
        self.patch_size_slider.pack(fill=tk.X)
        self.iterations_slider = tk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, label="Iterations")
        self.iterations_slider.set(5)
        self.iterations_slider.pack(fill=tk.X)

        tk.Label(control_frame, text="Actions", font=("Arial", 12, "bold")).pack(anchor="w", pady=(20, 0))
        self.btn_inpaint = tk.Button(control_frame, text="Inpaint", command=self.run_inpainting_thread, state=tk.DISABLED)
        self.btn_inpaint.pack(fill=tk.X, pady=5)
        self.btn_save_mask = tk.Button(control_frame, text="Save Mask", command=self.save_mask, state=tk.DISABLED)
        self.btn_save_mask.pack(fill=tk.X, pady=5)
        self.btn_toggle = tk.Button(control_frame, text="Toggle View", command=self.toggle_view, state=tk.DISABLED)
        self.btn_toggle.pack(fill=tk.X)
        
        self.status_label = tk.Label(control_frame, text="Load an image to start.", relief=tk.SUNKEN, anchor="w")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coords)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not path:
            return
        
        self.image_path = path
        self.original_image = cv2.imread(self.image_path)
        self.mask = Image.new("L", (self.original_image.shape[1], self.original_image.shape[0]), 0)
        
        self.update_canvas_image()
        self.btn_inpaint.config(state=tk.NORMAL)
        self.btn_save_mask.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)
        self.btn_toggle.config(state=tk.DISABLED)
        self.is_showing_result = False
        self.status_label.config(text="Draw to mask the area to remove.")

    def update_canvas_image(self):
        if self.original_image is None:
            return

        img_pil = Image.fromarray(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        
        red_overlay = Image.new("RGBA", img_pil.size, (255, 0, 0, 0))
        mask_pil = self.mask.convert("L")
        alpha_mask = mask_pil.copy()
        
        img_pil.paste(red_overlay, (0,0), alpha_mask)

        self.display_image = ImageTk.PhotoImage(img_pil)
        self.canvas.config(width=self.display_image.width(), height=self.display_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)

    def draw(self, event):
        if self.original_image is None or self.is_showing_result:
            return

        x, y = event.x, event.y
        brush_size = self.brush_slider.get()
        
        fill_color = 0 if self.erase_mode.get() else 255

        draw = ImageDraw.Draw(self.mask)
        if self.last_x and self.last_y:
            draw.line([(self.last_x, self.last_y), (x, y)], fill=fill_color, width=brush_size, joint="curve")
        draw.ellipse([(x - brush_size/2, y - brush_size/2), (x + brush_size/2, y + brush_size/2)], fill=fill_color)
        
        self.last_x, self.last_y = x, y
        self.update_canvas_image()

    def reset_coords(self, event):
        self.last_x, self.last_y = None, None

    def run_inpainting_thread(self):
        self.status_label.config(text="Inpainting... please wait.")
        self.master.update()
        thread = threading.Thread(target=self.run_inpainting)
        thread.start()

    def run_inpainting(self):
        patch_size = self.patch_size_slider.get()
        iterations = self.iterations_slider.get()
        
        mask_pil = np.array(self.mask)
        mask_for_algo = np.where(mask_pil > 127, 0, 1).astype(np.uint8)
        
        print(f"DEBUG: mask_for_algo unique values: {np.unique(mask_for_algo)}")
        print(f"DEBUG: patch_size: {patch_size}, iterations: {iterations}")

        try:
            inpaint_nnf = InpaintNNF(self.original_image, mask_for_algo, patch_w=patch_size, max_pm_iters=iterations)
            inpainted_images = inpaint_nnf.inpaint()
            self.inpainted_result = inpainted_images[-1]
            
            self.master.after(0, self.on_inpainting_complete)
            
        except Exception as e:
            print(f"Error: {e}")
            self.master.after(0, lambda: self.on_inpainting_error(e))

    def on_inpainting_complete(self):
        self.is_showing_result = True
        self.toggle_view() 
        self.btn_save.config(state=tk.NORMAL)
        self.btn_toggle.config(state=tk.NORMAL)
        self.status_label.config(text="Inpainting complete!")

    def on_inpainting_error(self, error):
        messagebox.showerror("Error", f"An error occurred during inpainting:\n{error}")
        self.status_label.config(text="Error during inpainting.")

    def toggle_view(self):
        if self.inpainted_result is None:
            return

        self.is_showing_result = not self.is_showing_result
        if self.is_showing_result:
            img_pil = Image.fromarray(cv2.cvtColor(self.inpainted_result, cv2.COLOR_BGR2RGB))
            self.display_image = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
            self.status_label.config(text="Showing inpainted result.")
        else:
            self.update_canvas_image()
            self.status_label.config(text="Showing original with mask.")

    def save_result(self):
        if self.inpainted_result is None:
            return
        
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if save_path:
            try:
                cv2.imwrite(save_path, self.inpainted_result)
                messagebox.showinfo("Success", f"Image saved to {save_path}")
                self.status_label.config(text=f"Result saved.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{e}")

    def save_mask(self):
        if self.mask is None:
            return
        
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if save_path:
            try:
                mask_inverted = Image.eval(self.mask, lambda x: 255 - x)
                mask_inverted.save(save_path)
                messagebox.showinfo("Success", f"Mask saved to {save_path}")
                self.status_label.config(text=f"Mask saved.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mask:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = InpaintingGUI(root)
    root.mainloop()
