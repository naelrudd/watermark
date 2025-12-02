# ============================================================
# PROGRAM WATERMARKING DIGITAL - LSB (Least Significant Bit)
# ============================================================
# Teknik: Digital Image Watermarking
# Metode: LSB (Least Significant Bit)
# Fitur: Evaluasi Imperceptibility, Capacity, Robustness, Security
# Dataset: 12 gambar dengan ukuran dan karakteristik berbeda
# ============================================================

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
from datetime import datetime
import json
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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
        
        Returns:
            watermark_text: teks watermark yang diekstrak
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
        """
        Peak Signal-to-Noise Ratio
        Semakin tinggi semakin baik (>30dB = excellent)
        """
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
        """
        Structural Similarity Index
        Semakin tinggi semakin baik (1.0 = identical)
        """
        if len(original.shape) == 3:
            original = np.dot(original[...,:3], [0.2989, 0.5870, 0.1140])
        if len(watermarked.shape) == 3:
            watermarked = np.dot(watermarked[...,:3], [0.2989, 0.5870, 0.1140])
        
        original = original.astype(np.float64)
        watermarked = watermarked.astype(np.float64)
        
        s = ssim(original, watermarked, data_range=255)
        return round(float(s), 4)
    
    @staticmethod
    def calculate_ncc(original, watermarked):
        """
        Normalized Cross-Correlation
        Semakin tinggi semakin baik (1.0 = perfect)
        """
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
        """
        Menghitung kapasitas penyimpanan watermark
        Capacity = (bits yang tertanam / bits maksimal) * 100%
        """
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
        """
        Bit Error Rate - Persentase karakter yang berbeda
        Semakin rendah semakin baik (0 = sempurna)
        """
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
        """
        Shannon Entropy - Mengukur randomness/keacakan
        Semakin tinggi semakin aman (max=8, >7 = good)
        """
        if len(watermark_text) == 0:
            return 0
        
        byte_counts = np.bincount([ord(c) for c in watermark_text], minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(watermark_text)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return round(entropy, 4)


# ============================================================
# 3. FUNGSI UNTUK MEMBUAT DATASET BERAGAM
# ============================================================

def create_diverse_dataset():
    """
    Membuat dataset gambar dengan ukuran, warna, dan karakteristik berbeda
    Total: 12 gambar dengan ukuran dari 200x200 hingga 768x576
    """
    dataset = []
    
    # Dataset 1: Solid Red
    img1 = Image.new('RGB', (200, 200), color='red')
    dataset.append(('Solid_Red_200x200', np.array(img1)))
    
    # Dataset 2: Solid Blue
    img2 = Image.new('RGB', (400, 300), color='blue')
    dataset.append(('Solid_Blue_400x300', np.array(img2)))
    
    # Dataset 3: Gradient
    img3_arr = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(300):
        img3_arr[i, :, 0] = int(255 * i / 300)
        img3_arr[i, :, 1] = int(128 * i / 300)
        img3_arr[i, :, 2] = 255 - int(255 * i / 300)
    dataset.append(('Gradient_300x300', np.array(img3_arr)))
    
    # Dataset 4: Checkerboard
    img4_arr = np.zeros((500, 500, 3), dtype=np.uint8)
    for i in range(500):
        for j in range(500):
            if (i // 50 + j // 50) % 2 == 0:
                img4_arr[i, j] = [255, 255, 255]
            else:
                img4_arr[i, j] = [0, 0, 0]
    dataset.append(('Checkerboard_500x500', np.array(img4_arr)))
    
    # Dataset 5: Random Noise
    img5_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    dataset.append(('Random_Noise_256x256', np.array(img5_arr)))
    
    # Dataset 6: Radial Gradient
    img6_arr = np.zeros((600, 400, 3), dtype=np.uint8)
    center_x, center_y = 300, 200
    max_distance = np.sqrt(center_x**2 + center_y**2)
    for i in range(600):
        for j in range(400):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            intensity = int(255 * (1 - distance / max_distance))
            img6_arr[i, j] = [intensity, 200, 255 - intensity]
    dataset.append(('Radial_Gradient_600x400', np.array(img6_arr)))
    
    # Dataset 7: Horizontal Lines
    img7_arr = np.zeros((350, 350, 3), dtype=np.uint8)
    for i in range(0, 350, 20):
        img7_arr[i:i+10, :] = [100, 150, 200]
    dataset.append(('Horizontal_Lines_350x350', np.array(img7_arr)))
    
    # Dataset 8: Vertical Lines
    img8_arr = np.zeros((480, 360, 3), dtype=np.uint8)
    for j in range(0, 360, 25):
        img8_arr[:, j:j+10] = [200, 100, 150]
    dataset.append(('Vertical_Lines_480x360', np.array(img8_arr)))
    
    # Dataset 9: Diagonal Pattern
    img9_arr = np.zeros((640, 480, 3), dtype=np.uint8)
    for i in range(640):
        for j in range(480):
            if (i + j) % 30 < 15:
                img9_arr[i, j] = [255, 128, 0]
            else:
                img9_arr[i, j] = [0, 128, 255]
    dataset.append(('Diagonal_Pattern_640x480', np.array(img9_arr)))
    
    # Dataset 10: Complex Pattern
    img10_arr = np.ones((512, 512, 3), dtype=np.uint8) * 200
    for i in range(512):
        for j in range(512):
            if (i - 256)**2 + (j - 256)**2 < 10000:
                img10_arr[i, j] = [255, 0, 0]
    img10_arr[100:200, 100:200] = [0, 255, 0]
    for i in range(50):
        img10_arr[400+i, 50:50+2*i] = [0, 0, 255]
    dataset.append(('Complex_Pattern_512x512', np.array(img10_arr)))
    
    # Dataset 11: Smooth Gradient Blue to Red
    img11_arr = np.zeros((320, 240, 3), dtype=np.uint8)
    for j in range(240):
        img11_arr[:, j, 0] = int(255 * j / 240)
        img11_arr[:, j, 2] = 255 - int(255 * j / 240)
    dataset.append(('Smooth_Gradient_320x240', np.array(img11_arr)))
    
    # Dataset 12: High Resolution Subtle Pattern
    img12_arr = np.zeros((768, 576, 3), dtype=np.uint8)
    for i in range(768):
        for j in range(576):
            img12_arr[i, j] = [
                int(100 + 50 * np.sin(i / 100)),
                int(100 + 50 * np.sin(j / 80)),
                150
            ]
    dataset.append(('High_Res_768x576', np.array(img12_arr)))
    
    return dataset


# ============================================================
# 4. FUNGSI UTAMA - EVALUASI LENGKAP
# ============================================================

def main():
    """
    Fungsi utama untuk menjalankan watermarking dan evaluasi
    """
    print("\n" + "="*80)
    print("PROGRAM DIGITAL IMAGE WATERMARKING - LSB METHOD")
    print("="*80)
    
    # 1. Create dataset
    print("\n[1] Membuat dataset gambar beragam...")
    dataset = create_diverse_dataset()
    print(f"    ✓ {len(dataset)} gambar telah dibuat\n")
    
    # 2. Prepare watermarks
    watermark_texts = [
        "Secret1", "Message2", "Data3", "Hidden", "Secure", "Water1",
        "Test01", "Mark08", "Sign99", "Info10", "Guard", "Protect"
    ]
    
    # 3. Process watermarking
    print("[2] Memproses watermarking pada semua dataset...\n")
    
    results = []
    watermarker = LSBWatermarking(seed=42)
    evaluator = WatermarkEvaluation()
    
    for idx, ((dataset_name, image_array), watermark_text) in enumerate(zip(dataset, watermark_texts), 1):
        print(f"    [{idx:2d}] Memproses: {dataset_name} (ukuran: {image_array.shape})")
        
        # Embedding
        watermarked, embedded_bits = watermarker.embed_watermark(image_array, watermark_text)
        extracted = watermarker.extract_watermark(watermarked)
        
        # Imperceptibility
        psnr = evaluator.calculate_psnr(image_array, watermarked)
        ssim_val = evaluator.calculate_ssim(image_array, watermarked)
        ncc = evaluator.calculate_ncc(image_array, watermarked)
        imperceptibility_score = (ssim_val + ncc) / 2
        
        # Capacity
        capacity_percent, watermark_bits, max_bits = evaluator.calculate_capacity(image_array.shape, watermark_text)
        
        # Robustness
        attacked_blur = evaluator.apply_gaussian_blur(watermarked, sigma=1.0)
        extracted_blur = watermarker.extract_watermark(attacked_blur)
        ber_blur = evaluator.calculate_ber(watermark_text, extracted_blur)
        
        attacked_noise = evaluator.apply_salt_pepper_noise(watermarked, density=0.01)
        extracted_noise = watermarker.extract_watermark(attacked_noise)
        ber_noise = evaluator.calculate_ber(watermark_text, extracted_noise)
        
        attacked_brightness = evaluator.apply_brightness_change(watermarked, factor=0.9)
        extracted_brightness = watermarker.extract_watermark(attacked_brightness)
        ber_brightness = evaluator.calculate_ber(watermark_text, extracted_brightness)
        
        robustness_score = 100 - ((ber_blur + ber_noise + ber_brightness) / 3)
        
        # Security
        entropy = evaluator.calculate_entropy(watermark_text)
        security_score = entropy / 8 * 100
        
        # Overall
        overall_score = (imperceptibility_score * 30 + robustness_score * 30 + security_score * 40) / 100
        
        results.append({
            'dataset_name': dataset_name,
            'watermark_text': watermark_text,
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
            'extraction_success': watermark_text == extracted
        })
    
    # 4. Create summary
    print(f"\n[3] Membuat ringkasan hasil...\n")
    
    df_results = pd.DataFrame(results)
    
    print("    "+"-"*80)
    print("    RINGKASAN STATISTIK")
    print("    "+"-"*80)
    print(f"    Imperceptibility (Avg PSNR): {df_results['psnr'].mean():.4f} dB")
    print(f"    Imperceptibility (Avg SSIM): {df_results['ssim'].mean():.4f}")
    print(f"    Capacity (Rata-rata): {df_results['capacity_percent'].mean():.6f}%")
    print(f"    Robustness Score: {df_results['robustness_score'].mean():.4f}")
    print(f"    Security Score: {df_results['security_score'].mean():.4f}")
    print(f"    ★ OVERALL SCORE: {df_results['overall_score'].mean():.4f}/100")
    print("    "+"-"*80)
    
    # 5. Save results
    csv_filename = 'watermark_evaluation_results.csv'
    df_results.to_csv(csv_filename, index=False)
    print(f"\n[4] Hasil telah disimpan ke: {csv_filename}\n")
    
    print("="*80)
    print("EVALUASI SELESAI - Program berhasil dijalankan!")
    print("="*80 + "\n")
    
    return df_results


if __name__ == "__main__":
    results_df = main()
