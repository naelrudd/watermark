# ============================================================
# PROGRAM WATERMARKING DIGITAL - LSB (Least Significant Bit)
# VERSION 3 - OTOMATIS 10 ITERASI BIT
# ============================================================
# Teknik: Digital Image Watermarking
# Metode: LSB (Least Significant Bit)
# Fitur: Otomatis generate 10 hasil evaluasi dan 10 gambar watermarked
# Output: 10 gambar + 1 file Excel dengan semua hasil metrik
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
from tkinter import scrolledtext

# ============================================================
# 1. KELAS WATERMARKING - LSB METHOD
# ============================================================

class LSBWatermarking:
    """
    Implementasi Digital Image Watermarking menggunakan metode LSB
    (Least Significant Bit) yang sederhana namun efektif.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def embed_watermark(self, image_array, watermark_text):
        """Menyisipkan watermark teks ke dalam gambar menggunakan LSB"""
        watermarked = image_array.copy().astype(np.uint32)
        
        if len(image_array.shape) == 2:
            watermarked = np.stack([watermarked, watermarked, watermarked], axis=2)
        
        watermark_bytes = watermark_text.encode('utf-8')
        watermark_bits = ''.join(format(byte, '08b') for byte in watermark_bytes)
        
        length = len(watermark_text)
        length_bits = format(length, '032b')
        full_bits = length_bits + watermark_bits
        
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
        """Mengekstrak watermark dari gambar"""
        watermarked = watermarked_array.astype(np.uint32)
        
        if len(watermarked.shape) == 2:
            watermarked = np.stack([watermarked, watermarked, watermarked], axis=2)
        
        extracted_bits = []
        for i in range(watermarked.shape[0]):
            for j in range(watermarked.shape[1]):
                for k in range(min(3, watermarked.shape[2])):
                    extracted_bits.append(str(watermarked[i, j, k] & 1))
        
        if len(extracted_bits) < 32:
            return "Error: Not enough bits"
        
        length_bits = ''.join(extracted_bits[:32])
        length = int(length_bits, 2)
        
        start_bit = 32
        end_bit = start_bit + (length * 8)
        
        if end_bit > len(extracted_bits):
            return "Error: Insufficient bits"
        
        watermark_bits = ''.join(extracted_bits[start_bit:end_bit])
        
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
    """Mengevaluasi kualitas watermark"""
    
    @staticmethod
    def calculate_psnr(original, watermarked):
        """Peak Signal-to-Noise Ratio"""
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
        """Structural Similarity Index"""
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
    
    @staticmethod
    def calculate_capacity(image_shape, watermark_text):
        """Menghitung kapasitas penyimpanan watermark"""
        watermark_bits = len(watermark_text) * 8 + 32
        
        if len(image_shape) == 3:
            max_bits = image_shape[0] * image_shape[1] * 3
        else:
            max_bits = image_shape[0] * image_shape[1] * 3
        
        capacity = (watermark_bits / max_bits) * 100
        return round(capacity, 4), watermark_bits, max_bits
    
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
        """Bit Error Rate"""
        if len(watermark_original) == 0:
            return 0.0
        
        if "Error" in str(watermark_recovered):
            return 100.0
        
        min_len = min(len(watermark_original), len(watermark_recovered))
        errors = sum(1 for i in range(min_len) if watermark_original[i] != watermark_recovered[i])
        ber = (errors / len(watermark_original)) * 100
        return round(ber, 4)
    
    @staticmethod
    def calculate_entropy(watermark_text):
        """Shannon Entropy"""
        if len(watermark_text) == 0:
            return 0
        
        byte_counts = np.bincount([ord(c) for c in watermark_text], minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(watermark_text)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return round(entropy, 4)


# ============================================================
# 3. FUNGSI PROSES 10 ITERASI OTOMATIS
# ============================================================

def process_10_iterations(image_path, watermark_text, output_folder="output_watermarked"):
    """
    Memproses watermarking untuk 10 iterasi bit secara otomatis.
    Menghasilkan:
    - 10 gambar watermarked (bit 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    - 1 file Excel dengan semua hasil evaluasi
    
    Args:
        image_path: path ke gambar input
        watermark_text: teks watermark
        output_folder: folder untuk menyimpan hasil
    
    Returns:
        list of dict: hasil evaluasi untuk setiap iterasi
        str: path ke file Excel hasil
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    
    # Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Baca gambar
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    image_array = np.array(img)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    watermarker = LSBWatermarking(seed=42)
    evaluator = WatermarkEvaluation()
    
    all_results = []
    
    # Proses 10 iterasi
    for iteration in range(1, 11):
        target_bits = 4 * (2 ** (iteration - 1))  # 4, 8, 16, 32, ..., 2048
        
        # Konversi teks ke bit
        full_bytes = watermark_text.encode('utf-8')
        full_bits = ''.join(format(b, '08b') for b in full_bytes)
        
        # Ambil prefix bit sesuai target
        prefix_bits = full_bits[:target_bits] if len(full_bits) >= target_bits else full_bits
        
        # Bentuk ulang watermark_text_truncated
        truncated_bytes = bytearray()
        for i in range(0, len(prefix_bits), 8):
            byte_bits = prefix_bits[i:i+8]
            if len(byte_bits) == 8:
                truncated_bytes.append(int(byte_bits, 2))
        
        if len(truncated_bytes) == 0:
            watermark_text_truncated = watermark_text[:1] if watermark_text else ""
        else:
            watermark_text_truncated = truncated_bytes.decode('utf-8', errors='ignore')
        
        # Embedding
        watermarked, embedded_bits = watermarker.embed_watermark(image_array, watermark_text_truncated)
        extracted = watermarker.extract_watermark(watermarked)
        
        # Evaluasi Imperceptibility
        psnr = evaluator.calculate_psnr(image_array, watermarked)
        ssim_val = evaluator.calculate_ssim(image_array, watermarked)
        ncc = evaluator.calculate_ncc(image_array, watermarked)
        imperceptibility_score = (ssim_val + ncc) / 2
        
        # Evaluasi Capacity
        capacity_percent, watermark_bits, max_bits = evaluator.calculate_capacity(
            image_array.shape, watermark_text_truncated
        )
        
        # Evaluasi Robustness
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
        
        # Evaluasi Security
        entropy = evaluator.calculate_entropy(watermark_text_truncated)
        security_score = entropy / 8 * 100
        
        # Overall Score
        overall_score = (imperceptibility_score * 30 + robustness_score * 30 + security_score * 40) / 100
        
        # Simpan gambar watermarked
        output_filename = f"{base_name}_watermarked_iter{iteration}_{target_bits}bit.png"
        output_path = os.path.join(output_folder, output_filename)
        
        watermarked_img = Image.fromarray(watermarked)
        watermarked_img.save(output_path)
        
        # Simpan hasil evaluasi
        result = {
            'Iteration': iteration,
            'Target_Bits': target_bits,
            'Filename': base_name,
            'Watermark_Used': watermark_text_truncated,
            'Image_Shape': str(image_array.shape),
            'PSNR': psnr,
            'SSIM': ssim_val,
            'NCC': ncc,
            'Imperceptibility_Score': round(imperceptibility_score, 4),
            'Capacity_Percent': capacity_percent,
            'Watermark_Bits': watermark_bits,
            'Max_Bits': max_bits,
            'BER_Blur': ber_blur,
            'BER_Noise': ber_noise,
            'BER_Brightness': ber_brightness,
            'Robustness_Score': round(robustness_score, 4),
            'Entropy': entropy,
            'Security_Score': round(security_score, 4),
            'Overall_Score': round(overall_score, 4),
            'Extraction_Success': watermark_text_truncated == extracted,
            'Extracted_Watermark': extracted,
            'Output_Path': output_path
        }
        
        all_results.append(result)
    
    # Simpan hasil ke Excel
    df = pd.DataFrame(all_results)
    excel_filename = f"{base_name}_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_path = os.path.join(output_folder, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Sheet ringkasan
        summary_df = df[['Iteration', 'Target_Bits', 'PSNR', 'SSIM', 'NCC', 
                         'Imperceptibility_Score', 'Robustness_Score', 
                         'Security_Score', 'Overall_Score']]
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    return all_results, excel_path


# ============================================================
# 4. GUI UNTUK UPLOAD GAMBAR & PROSES OTOMATIS
# ============================================================

class WatermarkAutoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Program - Auto 10 Iterations")
        self.root.geometry("1000x700")
        
        self.image_path = None
        self._build_widgets()
    
    def _build_widgets(self):
        # Frame atas: kontrol
        top = tk.Frame(self.root, pady=10, padx=10)
        top.pack(side=tk.TOP, fill=tk.X)
        
        btn_select = tk.Button(top, text="Pilih Gambar...", command=self.select_image, 
                               width=15, height=2)
        btn_select.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.lbl_image = tk.Label(top, text="Belum ada gambar yang dipilih", 
                                  font=("Arial", 10))
        self.lbl_image.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        tk.Label(top, text="Teks Watermark:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_text = tk.Entry(top, width=60, font=("Arial", 10))
        self.entry_text.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        tk.Label(top, text="Folder Output:", font=("Arial", 10, "bold")).grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_output = tk.Entry(top, width=60, font=("Arial", 10))
        self.entry_output.insert(0, "output_watermarked")
        self.entry_output.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        btn_process = tk.Button(top, text="Proses 10 Iterasi", command=self.process,
                               bg="#4CAF50", fg="white", width=20, height=2,
                               font=("Arial", 10, "bold"))
        btn_process.grid(row=3, column=0, columnspan=2, padx=5, pady=10)
        
        # Frame tengah: info hasil
        mid = tk.Frame(self.root, padx=10, pady=5)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        tk.Label(mid, text="Log Proses:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        self.text_log = scrolledtext.ScrolledText(mid, height=30, width=120, 
                                                  font=("Courier", 9))
        self.text_log.pack(fill=tk.BOTH, expand=True, pady=5)
    
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
        self.log(f"✓ Gambar dipilih: {os.path.basename(path)}")
    
    def log(self, message):
        """Menambahkan log ke text widget"""
        self.text_log.insert(tk.END, message + "\n")
        self.text_log.see(tk.END)
        self.root.update()
    
    def process(self):
        if not self.image_path:
            messagebox.showwarning("Peringatan", "Silakan pilih gambar terlebih dahulu.")
            return
        
        text = self.entry_text.get().strip()
        if not text:
            messagebox.showwarning("Peringatan", "Silakan isi teks watermark.")
            return
        
        output_folder = self.entry_output.get().strip()
        if not output_folder:
            output_folder = "output_watermarked"
        
        self.text_log.delete("1.0", tk.END)
        self.log("="*80)
        self.log("MEMULAI PROSES WATERMARKING - 10 ITERASI OTOMATIS")
        self.log("="*80)
        self.log(f"Gambar: {os.path.basename(self.image_path)}")
        self.log(f"Watermark: {text}")
        self.log(f"Output folder: {output_folder}")
        self.log("")
        
        try:
            results, excel_path = process_10_iterations(
                self.image_path, text, output_folder
            )
            
            self.log("\n" + "="*80)
            self.log("HASIL EVALUASI - 10 ITERASI")
            self.log("="*80)
            
            for r in results:
                self.log(f"\n--- Iterasi {r['Iteration']}: {r['Target_Bits']} bit ---")
                self.log(f"  Watermark dipakai : {r['Watermark_Used'][:50]}...")
                self.log(f"  PSNR              : {r['PSNR']:.4f} dB")
                self.log(f"  SSIM              : {r['SSIM']:.4f}")
                self.log(f"  NCC               : {r['NCC']:.4f}")
                self.log(f"  Imperceptibility  : {r['Imperceptibility_Score']:.4f}")
                self.log(f"  Capacity          : {r['Capacity_Percent']:.6f}%")
                self.log(f"  Robustness        : {r['Robustness_Score']:.4f}")
                self.log(f"  Security          : {r['Security_Score']:.4f}")
                self.log(f"  Overall Score     : {r['Overall_Score']:.4f}")
                self.log(f"  Extract Success   : {r['Extraction_Success']}")
                self.log(f"  Output            : {os.path.basename(r['Output_Path'])}")
            
            self.log("\n" + "="*80)
            self.log("✓ PROSES SELESAI!")
            self.log("="*80)
            self.log(f"\n10 gambar watermarked telah disimpan di folder: {output_folder}")
            self.log(f"File Excel evaluasi: {os.path.basename(excel_path)}")
            self.log("")
            
            messagebox.showinfo(
                "Sukses", 
                f"Proses selesai!\n\n"
                f"10 gambar watermarked dan 1 file Excel\n"
                f"telah disimpan di folder: {output_folder}"
            )
            
        except Exception as e:
            self.log(f"\n❌ ERROR: {str(e)}")
            messagebox.showerror("Error", f"Terjadi kesalahan:\n{e}")


def main():
    root = tk.Tk()
    app = WatermarkAutoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()