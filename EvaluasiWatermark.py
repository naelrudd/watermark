import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# =======================
# Fungsi Metrik
# =======================
def calculate_psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf') 
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return round(psnr_value, 4)

def calculate_ssim(original, watermarked):
    if len(original.shape) == 3 and original.shape[2] == 3:
        ssim_value = ssim(original, watermarked, multichannel=True)
    else:
        ssim_value = ssim(original, watermarked)
    return round(ssim_value, 4)

def calculate_ber(original, watermarked):
    original_bits = np.unpackbits(original.astype(np.uint8))
    watermarked_bits = np.unpackbits(watermarked.astype(np.uint8))
    ber_value = np.mean(original_bits != watermarked_bits)
    return round(ber_value, 4)

def calculate_nc(original, watermarked):
    # Normalize images to [0, 1] range
    original_norm = original.astype(np.float64) / 255.0
    watermarked_norm = watermarked.astype(np.float64) / 255.0
    
    # Calculate normalized correlation
    numerator = np.sum(original_norm * watermarked_norm)
    denominator = np.sqrt(np.sum(original_norm ** 2) * np.sum(watermarked_norm ** 2))
    nc_value = numerator / denominator if denominator != 0 else 0
    
    # Ensure the result is in [0, 1] range
    nc_value = abs(nc_value)
    return round(nc_value, 4)

# =======================
# Tooltip sederhana
# =======================
class CreateToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

# =======================
# GUI
# =======================
class WatermarkEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Evaluator Modern")
        self.root.geometry("700x700")
        self.root.minsize(650, 450)
        self.root.configure(bg="#f5f5f5")

        self.original_image_path = None
        self.watermarked_image_path = None
        self.original_thumbnail = None
        self.watermarked_thumbnail = None

        # ======= Bagian Atas: Judul =======
        title_frame = tk.Frame(root, bg="#f5f5f5")
        title_frame.pack(fill="x", pady=10)
        title_label = tk.Label(title_frame, text="Watermark Evaluator", font=("Helvetica", 24, "bold"), bg="#f5f5f5", fg="#333")
        title_label.pack()

        # ======= Bagian Tengah: Pemilihan Gambar =======
        middle_frame = tk.Frame(root, bg="#f5f5f5", padx=20, pady=10)
        middle_frame.pack(fill="x")

        # Original Image
        orig_frame = tk.Frame(middle_frame, bg="#f5f5f5", pady=5)
        orig_frame.grid(row=0, column=0, padx=20, sticky="n")
        self.btn_original = ttk.Button(orig_frame, text="Browse Original Image", command=self.load_original)
        self.btn_original.pack()
        CreateToolTip(self.btn_original, "Pilih gambar asli dari komputer")
        self.label_original_name = tk.Label(orig_frame, text="Belum ada file", bg="#f5f5f5")
        self.label_original_name.pack(pady=5)
        self.label_original_img = tk.Label(orig_frame, bg="#f5f5f5")
        self.label_original_img.pack()

        # Watermarked Image
        water_frame = tk.Frame(middle_frame, bg="#f5f5f5", pady=5)
        water_frame.grid(row=0, column=1, padx=20, sticky="n")
        self.btn_watermarked = ttk.Button(water_frame, text="Browse Watermarked Image", command=self.load_watermarked)
        self.btn_watermarked.pack()
        CreateToolTip(self.btn_watermarked, "Pilih gambar hasil watermark dari komputer")
        self.label_watermarked_name = tk.Label(water_frame, text="Belum ada file", bg="#f5f5f5")
        self.label_watermarked_name.pack(pady=5)
        self.label_watermarked_img = tk.Label(water_frame, bg="#f5f5f5")
        self.label_watermarked_img.pack()

        # ======= Bagian Bawah: Tombol Hitung dan Hasil =======
        bottom_frame = tk.Frame(root, bg="#f5f5f5", padx=20, pady=10)
        bottom_frame.pack(fill="both", expand=True)

        self.btn_calculate = tk.Button(bottom_frame, text="Hitung Metrik", command=self.calculate_metrics,
                                       bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), relief="raised")
        self.btn_calculate.pack(pady=10)
        CreateToolTip(self.btn_calculate, "Klik untuk menghitung PSNR, SSIM, BER, dan NC")

        # Hasil
        result_frame = tk.LabelFrame(bottom_frame, text="Hasil Perhitungan", padx=20, pady=10, font=("Helvetica", 12, "bold"))
        result_frame.pack(fill="both", expand=True, pady=10)

        self.result_labels = {}
        for i, metric in enumerate(["PSNR", "SSIM", "BER", "NC"]):
            tk.Label(result_frame, text=f"{metric}:", anchor="w", font=("Courier", 12, "bold")).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            lbl = tk.Label(result_frame, text="-", anchor="w", font=("Courier", 12, "bold"))
            lbl.grid(row=i, column=1, sticky="w", padx=5, pady=5)
            self.result_labels[metric] = lbl

        # Responsive grid
        for i in range(2):
            middle_frame.columnconfigure(i, weight=1)
        bottom_frame.columnconfigure(0, weight=1)

    # =======================
    # Fungsi Tombol
    # =======================
    def load_original(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.bmp *.jpeg")])
        if path:
            self.original_image_path = path
            self.label_original_name.config(text=path.split("/")[-1])
            self.show_thumbnail(path, "original")

    def load_watermarked(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.bmp *.jpeg")])
        if path:
            self.watermarked_image_path = path
            self.label_watermarked_name.config(text=path.split("/")[-1])
            self.show_thumbnail(path, "watermarked")

    def show_thumbnail(self, path, img_type):
        img = Image.open(path)
        img.thumbnail((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        if img_type == "original":
            self.original_thumbnail = img_tk
            self.label_original_img.config(image=self.original_thumbnail)
        else:
            self.watermarked_thumbnail = img_tk
            self.label_watermarked_img.config(image=self.watermarked_thumbnail)

    def calculate_metrics(self):
        if not self.original_image_path or not self.watermarked_image_path:
            messagebox.showerror("Error", "Kedua gambar harus dipilih!")
            return

        original = cv2.imread(self.original_image_path, cv2.IMREAD_GRAYSCALE)
        watermarked = cv2.imread(self.watermarked_image_path, cv2.IMREAD_GRAYSCALE)

        if original.shape != watermarked.shape:
            watermarked = cv2.resize(watermarked, (original.shape[1], original.shape[0]))

        psnr_val = calculate_psnr(original, watermarked)
        ssim_val = calculate_ssim(original, watermarked)
        ber_val = calculate_ber(original, watermarked)
        nc_val = calculate_nc(original, watermarked)

        self.result_labels["PSNR"].config(text=str(psnr_val))
        self.result_labels["SSIM"].config(text=str(ssim_val))
        self.result_labels["BER"].config(text=str(ber_val))
        self.result_labels["NC"].config(text=str(nc_val))

# =======================
# Jalankan Aplikasi
# =======================
if __name__ == "__main__":
    root = tk.Tk()
    app = WatermarkEvaluatorApp(root)
    root.mainloop()
