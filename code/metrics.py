import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_psnr(original, reconstructed):
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, reconstructed):
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if len(reconstructed.shape) == 3:
        reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
        
    return ssim(original, reconstructed, data_range=original.max() - original.min())

def calculate_mse(original, reconstructed):
    return np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
