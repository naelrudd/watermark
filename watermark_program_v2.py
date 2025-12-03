# ============================================================
# PROGRAM WATERMARKING DIGITAL - LSB (Least Significant Bit)
# VERSION 2 - BACA DARI FOLDER "gambar"
# ============================================================
# Teknik: Digital Image Watermarking
# Metode: LSB (Least Significant Bit)
# Fitur: Evaluasi Imperceptibility, Capacity, Robustness, Security
# Dataset: Gambar dari folder "gambar" (user dapat upload sendiri)
# ============================================================

import numpy as np
from PIL import Image, ImageTk
import os
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# ============================================================
# 1. KELAS WATERMARKING - LSB METHOD
# ============================================================

class LSBWatermarking:
    """
    Implementasi Digital Image Watermarking menggunakan metode LSB
    (Least Significant Bit) yang sederhana namun efektif.
    
    Prinsip: Menyisipkan data ke dalam bit paling rendah pixel gambar
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def embed_watermark(self, image_array, watermark_text):
        """
        Menyisipkan watermark teks ke dalam gambar menggunakan LSB
        
        Args:
            image_array: numpy array dari gambar asli (RGB)
            watermark_text: string yang akan disembunyikan
        
        Returns:
            watermarked_image: gambar dengan watermark
            embedded_count: jumlah bit yang tertanam
        """
        watermarked = image_array.copy().astype(np.uint32)
        
        # Handle grayscale atau RGB
        if len(image_array.shape) == 2:
            watermarked = np.stack([watermarked, watermarked, watermarked], axis=2)
        
        # Encode watermark
        watermark_bytes = watermark_text.encode('utf-8')
        watermark_bits = ''.join(format(byte, '08b') for byte in watermark_bytes)
        
        # Add length header (32 bits)
        length = len(watermark_text)
        length_bits = format(length, '032b')
        full_bits = length_bits + watermark_bits
        
        # Embed ke LSB
        bit_index = 0
        embedded_count = 0
        
        for i in range(watermarked.shape[0]):
            for j in range(watermarked.shape[1]):
                for k in range(min(3, watermarked.shape[2])):
                    if bit_index < len(full_bits):
                        watermarked[i, j, k] = (watermarked[i, j, k] & 0xFFFFFFFE) | int(full_bits[bit_index])
                        bit_index += 1
                        embedded_count += 1
                    else:
                        break
                if bit_index >= len(full_bits):
                    break
            if bit_index >= len(full_bits):
                break
        
        return watermarked.astype(np.uint8), embedded_count
    
    def extract_watermark(self, watermarked_array):
        """
        Mengekstrak watermark dari gambar
        """
        watermarked = watermarked_array.astype(np.uint32)
        
        # Handle grayscale atau RGB
        if len(watermarked.shape) == 2:
            watermarked = np.stack([watermarked, watermarked, watermarked], axis=2)
        
        # Extract LSB
        extracted_bits = []
        for i in range(watermarked.shape[0]):
            for j in range(watermarked.shape[1]):
                for k in range(min(3, watermarked.shape[2])):
                    extracted_bits.append(str(watermarked[i, j, k] & 1))
        
        # Extract length (first 32 bits)
        if len(extracted_bits) < 32:
            return "Error: Not enough bits"
        
        length_bits = ''.join(extracted_bits[:32])
        length = int(length_bits, 2)
        
        # Extract watermark
        start_bit = 32
        end_bit = start_bit + (length * 8)
        
        if end_bit > len(extracted_bits):
            return "Error: Insufficient bits"
        
        watermark_bits = ''.join(extracted_bits[start_bit:end_bit])
        
        # Convert bits to bytes
        watermark_bytes = bytearray()
        for i in range(0, len(watermark_bits), 8):
            byte_bits = watermark_bits[i:i+8]
            if len(byte_bits) == 8:
                watermark_bytes.append(int(byte_bits, 2))
        
        return watermark_bytes.decode('utf-8', errors='replace')


# ============================================================
# 2. KELAS EVALUASI METRIK
# ============================================================

class WatermarkEvaluation:
    """
    Mengevaluasi kualitas watermark berdasarkan 4 kriteria utama:
    1. IMPERCEPTIBILITY - Kualitas visual (PSNR, SSIM, NCC)
    2. CAPACITY - Kapasitas penyimpanan data
    3. ROBUSTNESS - Ketahanan terhadap serangan
    4. SECURITY - Keamanan watermark
    """
    
    # --------- IMPERCEPTIBILITY (Kualitas Visual) ---------
    @staticmethod
    def calculate_psnr(original, watermarked):
        """Peak Signal-to-Noise Ratio - Semakin tinggi semakin baik (>30dB baik)"""
        # Convert to grayscale if RGB
        if len(original.shape) == 3:
            original = np.dot(original[...,:3], [0.2989, 0.5870, 0.1140])
        if len(watermarked.shape) == 3:
            watermarked = np.dot(watermarked[...,:3], [0.2989, 0.5870, 0.1140])
        
        original = original.astype(np.float64)
        watermarked = watermarked.astype(np.float64)
        
        mse = np.mean((original - watermarked) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return round(float(psnr), 4)
    
    @staticmethod
    def calculate_ssim(original, watermarked):
        """Structural Similarity Index - Semakin tinggi semakin baik (1.0 ideal)"""
        # Convert to grayscale if RGB
        if len(original.shape) == 3:
            original = np.dot(original[...,:3], [0.2989, 0.5870, 0.1140])
        if len(watermarked.shape) == 3:
            watermarked = np.dot(watermarked[...,:3], [0.2989, 0.5870, 0.1140])
        
        original = original.astype(np.float64)
        watermarked = watermarked.astype(np.float64)
        
        s = ssim(original, watermarked, data_range=255, full=False)
        if isinstance(s, tuple):
            s = s[0]
        return round(float(s), 4)
    
    @staticmethod
    def calculate_ncc(original, watermarked):
        """Normalized Cross-Correlation"""
        if len(original.shape) == 3:
            original = np.dot(original[...,:3], [0.2989, 0.5870, 0.1140])
        if len(watermarked.shape) == 3:
            watermarked = np.dot(watermarked[...,:3], [0.2989, 0.5870, 0.1140])
        
        original = original.astype(np.float64).flatten()
        watermarked = watermarked.astype(np.float64).flatten()
        
        mean_orig = np.mean(original)
        mean_water = np.mean(watermarked)
        
        std_orig = np.std(original)
        std_water = np.std(watermarked)
        
        if std_orig == 0 or std_water == 0:
            return 1.0 if np.array_equal(original, watermarked) else 0.0
        
        ncc = np.mean((original - mean_orig) * (watermarked - mean_water)) / (std_orig * std_water)
        return round(float(abs(ncc)), 4)
    
    # --------- CAPACITY (Kapasitas Data) ---------
    @staticmethod
    def calculate_capacity(image_shape, watermark_text):
        """Menghitung kapasitas penyimpanan watermark"""
        watermark_bits = len(watermark_text) * 8 + 32  # +32 untuk length header
        
        if len(image_shape) == 3:
            max_bits = image_shape[0] * image_shape[1] * 3  # 3 channels RGB
        else:
            max_bits = image_shape[0] * image_shape[1] * 3
        
        capacity = (watermark_bits / max_bits) * 100
        return round(capacity, 4), watermark_bits, max_bits
    
    # --------- ROBUSTNESS (Ketahanan terhadap Serangan) ---------
    @staticmethod
    def apply_gaussian_blur(image, sigma=1.0):
        """Serangan: Gaussian Blur"""
        result = np.zeros_like(image, dtype=np.float32)
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                result[:,:,c] = gaussian_filter(image[:,:,c].astype(np.float32), sigma=sigma)
        else:
            result = gaussian_filter(image.astype(np.float32), sigma=sigma)
        return result.astype(np.uint8)
    
    @staticmethod
    def apply_salt_pepper_noise(image, density=0.01):
        """Serangan: Salt & Pepper Noise"""
        noisy = image.copy().astype(np.float32)
        num_pixels = int(image.size * density)
        
        for _ in range(num_pixels):
            if len(image.shape) == 3:
                x = np.random.randint(0, image.shape[0])
                y = np.random.randint(0, image.shape[1])
                c = np.random.randint(0, image.shape[2])
                noisy[x, y, c] = 255 if np.random.rand() > 0.5 else 0
            else:
                x = np.random.randint(0, image.shape[0])
                y = np.random.randint(0, image.shape[1])
                noisy[x, y] = 255 if np.random.rand() > 0.5 else 0
        
        return noisy.astype(np.uint8)
    
    @staticmethod
    def apply_brightness_change(image, factor=0.9):
        """Serangan: Brightness Change"""
        adjusted = (image.astype(np.float32) * factor).astype(np.uint8)
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    @staticmethod
    def calculate_ber(watermark_original, watermark_recovered):
        """Bit Error Rate - Persentase karakter yang berbeda"""
        if len(watermark_original) == 0:
            return 0.0
        
        if "Error" in str(watermark_recovered):
            return 100.0
        
        min_len = min(len(watermark_original), len(watermark_recovered))
        errors = sum(1 for i in range(min_len) if watermark_original[i] != watermark_recovered[i])
        ber = (errors / len(watermark_original)) * 100
        return round(ber, 4)
    
    # --------- SECURITY (Keamanan) ---------
    @staticmethod
    def calculate_entropy(watermark_text):
        """Shannon Entropy - Mengukur randomness/keacakan"""
        if len(watermark_text) == 0:
            return 0
        
        byte_counts = np.bincount([ord(c) for c in watermark_text], minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(watermark_text)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return round(entropy, 4)


# ============================================================
# 3. FUNGSI EMBED & EVALUASI UNTUK SATU GAMBAR
# ============================================================

def process_single_image(image_path, watermark_text, bit_iteration=1, output_path=None):
    """
    Memproses watermarking dan evaluasi untuk SATU gambar.
    
    Args:
        image_path (str): path ke gambar input.
        watermark_text (str): teks watermark.
        bit_iteration (int): urutan ke berapa (1..10) untuk panjang bit watermark.
                             1 -> 4 bit, 2 -> 8 bit, 3 -> 16 bit, dst (kelipatan 2).
        output_path (str): path output gambar watermarked (opsional).
    Returns:
        dict: hasil metrik dan informasi penting lain.
        np.ndarray: array gambar watermarked.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    # Baca gambar
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    image_array = np.array(img)

    # Aturan panjang bit watermark: 4, 8, 16, 32, ... sampai iterasi ke-10
    if bit_iteration < 1:
        bit_iteration = 1
    if bit_iteration > 10:
        bit_iteration = 10

    # Konversi teks watermark ke bit
    full_bytes = watermark_text.encode('utf-8')
    full_bits = ''.join(format(b, '08b') for b in full_bytes)

    target_bits = 4 * (2 ** (bit_iteration - 1))  # 4, 8, 16, 32, ...
    # Ambil prefix bit sesuai target (kalau teks kurang panjang, pakai semua yg ada)
    prefix_bits = full_bits[:target_bits] if len(full_bits) >= target_bits else full_bits

    # Bentuk ulang watermark_text_truncated dari prefix_bits agar bisa diekstrak normal
    truncated_bytes = bytearray()
    for i in range(0, len(prefix_bits), 8):
        byte_bits = prefix_bits[i:i+8]
        if len(byte_bits) == 8:
            truncated_bytes.append(int(byte_bits, 2))

    if len(truncated_bytes) == 0:
        watermark_text_truncated = watermark_text[:1] if watermark_text else ""
    else:
        watermark_text_truncated = truncated_bytes.decode('utf-8', errors='ignore')

    # Embedding dengan teks hasil pemotongan sesuai aturan bit
    watermarker = LSBWatermarking(seed=42)
    evaluator = WatermarkEvaluation()

    watermarked, embedded_bits = watermarker.embed_watermark(image_array, watermark_text_truncated)
    extracted = watermarker.extract_watermark(watermarked)

    # Imperceptibility
    psnr = evaluator.calculate_psnr(image_array, watermarked)
    ssim_val = evaluator.calculate_ssim(image_array, watermarked)
    ncc = evaluator.calculate_ncc(image_array, watermarked)
    imperceptibility_score = (ssim_val + ncc) / 2

    # Capacity
    capacity_percent, watermark_bits, max_bits = evaluator.calculate_capacity(image_array.shape, watermark_text_truncated)

    # Robustness
    attacked_blur = evaluator.apply_gaussian_blur(watermarked, sigma=1.0)
    extracted_blur = watermarker.extract_watermark(attacked_blur)
    ber_blur = evaluator.calculate_ber(watermark_text_truncated, extracted_blur)

    attacked_noise = evaluator.apply_salt_pepper_noise(watermarked, density=0.01)
    extracted_noise = watermarker.extract_watermark(attacked_noise)
    ber_noise = evaluator.calculate_ber(watermark_text_truncated, extracted_noise)

    attacked_brightness = evaluator.apply_brightness_change(watermarked, factor=0.9)
    extracted_brightness = watermarker.extract_watermark(attacked_brightness)
    ber_brightness = evaluator.calculate_ber(watermark_text_truncated, extracted_brightness)

    robustness_score = 100 - ((ber_blur + ber_noise + ber_brightness) / 3)

    # Security
    entropy = evaluator.calculate_entropy(watermark_text_truncated)
    security_score = entropy / 8 * 100

    # Overall
    overall_score = (imperceptibility_score * 30 + robustness_score * 30 + security_score * 40) / 100

    # Simpan gambar watermarked
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"watermarked_{base_name}.png"

    # Hapus jika sudah ada versi lama
    if os.path.exists(output_path):
        os.remove(output_path)

    watermarked_img = Image.fromarray(watermarked)
    watermarked_img.save(output_path)

    result = {
        'filename': os.path.basename(image_path),
        'watermark_text': watermark_text_truncated,
        'original_watermark_text': watermark_text,
        'bit_iteration': bit_iteration,
        'target_bits_rule': target_bits,
        'image_shape': str(image_array.shape),
        'psnr': psnr,
        'ssim': ssim_val,
        'ncc': ncc,
        'imperceptibility_score': imperceptibility_score,
        'capacity_percent': capacity_percent,
        'ber_blur': ber_blur,
        'ber_noise': ber_noise,
        'ber_brightness': ber_brightness,
        'robustness_score': robustness_score,
        'entropy': entropy,
        'security_score': security_score,
        'overall_score': overall_score,
        'extraction_success': watermark_text == extracted,
        'embedded_bits': embedded_bits,
        'watermark_bits': watermark_bits,
        'max_bits': max_bits,
        'output_path': output_path,
        'extracted_watermark': extracted,
    }

    return result, watermarked


# ============================================================
# 4. UTILITAS CETAK RINGKASAN
# ============================================================

def _print_summary(result):
    """Cetak ringkasan hasil ke terminal (dipakai CLI atau debug)."""
    print("\n[HASIL]")
    print("-" * 80)
    print(f" Gambar          : {result['filename']}")
    print(f" Ukuran          : {result['image_shape']}")
    print(f" PSNR            : {result['psnr']:.4f} dB")
    print(f" SSIM            : {result['ssim']:.4f}")
    print(f" NCC             : {result['ncc']:.4f}")
    print(f" Capacity        : {result['capacity_percent']:.6f}%")
    print(f" Robustness      : {result['robustness_score']:.4f}")
    print(f" Security Score  : {result['security_score']:.4f}")
    print(f" Overall Score   : {result['overall_score']:.4f}")
    print(f" Extract Success : {result['extraction_success']}")
    print(f" Output Image    : {result['output_path']}")
    print("-" * 80 + "\n")


# ============================================================
# 3. FUNGSI UNTUK MEMBACA GAMBAR DARI FOLDER
# ============================================================

def load_images_from_folder(folder_path):
    """
    Membaca semua gambar dari folder
    
    Args:
        folder_path: path ke folder yang berisi gambar
    
    Returns:
        list of tuples (filename, image_array)
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    images = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' tidak ditemukan!")
        print(f"   Silahkan buat folder '{folder_path}' dan masukkan gambar di dalamnya")
        return []
    
    # Baca semua file di folder
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith(supported_formats)]
    
    if len(image_files) == 0:
        print(f"⚠️  Tidak ada gambar di folder '{folder_path}'")
        print(f"   Format yang didukung: {', '.join(supported_formats)}")
        return []
    
    print(f"✓ Ditemukan {len(image_files)} gambar di folder '{folder_path}':\n")
    
    for idx, filename in enumerate(sorted(image_files), 1):
        try:
            filepath = os.path.join(folder_path, filename)
            img = Image.open(filepath)
            
            # Convert ke RGB jika grayscale
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            images.append((filename, img_array))
            
            print(f"  {idx}. {filename:<30} Ukuran: {img_array.shape}")
        
        except Exception as e:
            print(f"  ❌ Error membaca {filename}: {str(e)}")
    
    print()
    return images


# ============================================================
# 4. FUNGSI UTAMA (MODE CLI SEDERHANA, SATU GAMBAR)
# ============================================================

def main_cli():
    """
    Mode terminal (opsional) untuk menjalankan watermarking dan evaluasi
    pada SATU gambar.
    """
    print("\n" + "="*80)
    print("PROGRAM DIGITAL IMAGE WATERMARKING - LSB METHOD (SINGLE IMAGE - CLI)")
    print("="*80)

    image_path = input("Masukkan path gambar: ").strip().strip('\"')
    watermark_text = input("Masukkan teks watermark: ").strip()

    if not image_path:
        print("⚠️  Path gambar kosong, program berhenti.")
        return
    if watermark_text == "":
        print("⚠️  Teks watermark kosong, program berhenti.")
        return

    try:
        # Default iterasi ke-1 (4 bit) untuk mode CLI
        result, _ = process_single_image(image_path, watermark_text, bit_iteration=1)
    except Exception as e:
        print(f"❌ Terjadi error: {e}")
        return

    _print_summary(result)


# ============================================================
# 5. GUI UNTUK UPLOAD GAMBAR & PREVIEW WATERMARK
# ============================================================

class WatermarkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Program 2 - Single Image LSB")
        self.root.geometry("980x620")

        # Hapus file watermarked lama di folder kerja
        self._cleanup_old_watermarked()

        self.image_path = None
        self.preview_original = None
        self.preview_watermarked = None

        self._build_widgets()

    def _cleanup_old_watermarked(self):
        """Hapus file yang diawali 'watermarked_' di folder saat ini."""
        cwd = os.getcwd()
        for fname in os.listdir(cwd):
            if fname.lower().startswith("watermarked_"):
                try:
                    os.remove(os.path.join(cwd, fname))
                except Exception:
                    pass

    def _build_widgets(self):
        # Frame atas: kontrol
        top = tk.Frame(self.root, pady=8, padx=8)
        top.pack(side=tk.TOP, fill=tk.X)

        btn_select = tk.Button(top, text="Pilih Gambar...", command=self.select_image)
        btn_select.grid(row=0, column=0, padx=4, pady=4, sticky="w")

        self.lbl_image = tk.Label(top, text="Belum ada gambar yang dipilih")
        self.lbl_image.grid(row=0, column=1, padx=4, pady=4, sticky="w")

        tk.Label(top, text="Teks Watermark:").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        self.entry_text = tk.Entry(top, width=50)
        self.entry_text.grid(row=1, column=1, padx=4, pady=4, sticky="w")

        tk.Label(top, text="Iterasi Bit (4,8,16,32,...):").grid(row=2, column=0, padx=4, pady=4, sticky="w")
        self.bit_var = tk.StringVar(value="1 - 4 bit")
        options = [f"{i} - {4 * (2 ** (i - 1))} bit" for i in range(1, 11)]
        self.combo_bits = ttk.Combobox(top, textvariable=self.bit_var, values=options, width=20, state="readonly")
        self.combo_bits.grid(row=2, column=1, padx=4, pady=4, sticky="w")

        btn_process = tk.Button(top, text="Proses Watermark", command=self.process)
        btn_process.grid(row=3, column=0, padx=4, pady=8, sticky="w")

        # Frame tengah: info hasil
        mid = tk.Frame(self.root, padx=8, pady=4)
        mid.pack(side=tk.TOP, fill=tk.X)

        self.text_info = tk.Text(mid, height=6, width=120)
        self.text_info.pack(fill=tk.X)
        self.text_info.configure(state="disabled")

        # Frame bawah: preview gambar
        bottom = tk.Frame(self.root, padx=8, pady=8)
        bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = tk.Frame(bottom)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(bottom)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left, text="Gambar Asli").pack()
        self.lbl_preview_original = tk.Label(left, bg="#e0e0e0")
        self.lbl_preview_original.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        tk.Label(right, text="Gambar Watermarked").pack()
        self.lbl_preview_watermarked = tk.Label(right, bg="#e0e0e0")
        self.lbl_preview_watermarked.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Pilih gambar",
            filetypes=(
                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return
        self.image_path = path
        self.lbl_image.config(text=os.path.basename(path))
        self._show_original_preview(path)

    def _show_original_preview(self, path):
        try:
            img = Image.open(path)
            img = img.convert("RGB")
            img = self._resize_for_preview(img)
            self.preview_original = ImageTk.PhotoImage(img)
            self.lbl_preview_original.config(image=self.preview_original)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat gambar asli:\n{e}")

    def _show_watermarked_preview(self, path):
        try:
            img = Image.open(path)
            img = img.convert("RGB")
            img = self._resize_for_preview(img)
            self.preview_watermarked = ImageTk.PhotoImage(img)
            self.lbl_preview_watermarked.config(image=self.preview_watermarked)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat gambar watermarked:\n{e}")

    @staticmethod
    def _resize_for_preview(img, max_size=(420, 320)):
        img.thumbnail(max_size, Image.LANCZOS)
        return img

    def process(self):
        if not self.image_path:
            messagebox.showwarning("Peringatan", "Silakan pilih gambar terlebih dahulu.")
            return

        text = self.entry_text.get().strip()
        if not text:
            messagebox.showwarning("Peringatan", "Silakan isi teks watermark.")
            return

        try:
            iter_str = self.bit_var.get().split(" - ")[0].strip()
            bit_iteration = int(iter_str)
        except Exception:
            bit_iteration = 1

        try:
            result, _ = process_single_image(self.image_path, text, bit_iteration=bit_iteration)
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat proses watermark:\n{e}")
            return

        # Tampilkan info hasil
        info_lines = [
            f"Gambar              : {result['filename']}",
            f"Ukuran              : {result['image_shape']}",
            f"Original Text       : {result.get('original_watermark_text', '')}",
            f"Watermark Dipakai   : {result['watermark_text']}",
            f"Iterasi Bit         : {result.get('bit_iteration', '-')}",
            f"Target Bits (aturan): {result.get('target_bits_rule', '-')}",
            f"PSNR                : {result['psnr']:.4f} dB",
            f"SSIM                : {result['ssim']:.4f}",
            f"NCC                 : {result['ncc']:.4f}",
            f"Capacity            : {result['capacity_percent']:.6f}%",
            f"Robustness Score    : {result['robustness_score']:.4f}",
            f"Security Score      : {result['security_score']:.4f}",
            f"Overall Score       : {result['overall_score']:.4f}",
            f"Extract Success     : {result['extraction_success']}",
            f"Output Image        : {result['output_path']}",
        ]

        self.text_info.configure(state="normal")
        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(tk.END, "\n".join(info_lines))
        self.text_info.configure(state="disabled")

        # Preview gambar watermarked
        self._show_watermarked_preview(result['output_path'])


def main_gui():
    root = tk.Tk()
    app = WatermarkGUI(root)
    root.mainloop()




if __name__ == "__main__":
    # Jalankan GUI sebagai default
    main_gui()
