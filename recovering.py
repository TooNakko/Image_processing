import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d


def box_blur(image, kernel_size):
    # Tạo kernel có giá trị là 1
    kernel_value = np.ones((kernel_size, kernel_size))
    
    # Chuẩn hóa kernel để tổng giá trị bằng 1 (đối với box blur)
    kernel_value /= np.sum(kernel_value)
    
    # Áp dụng convolution
    blurred_image = convolve2d(image, kernel_value, mode='same', boundary='symm')
    
    return blurred_image

def wiener_filter(img, kernel_size, K):
    
    # Normalize the kernel 
    normalized_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    # Pad the kernel to match the size of the image
    padded_kernel = np.pad(normalized_kernel, [(0, img.shape[0] - normalized_kernel.shape[0]), (0, img.shape[1] - normalized_kernel.shape[1])], 'constant')

    # Fourier Transform of the image and kernel
    img_fft = fft2(img)
    kernel_fft = fft2(padded_kernel)

    # Wiener filter formula
    kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    img_fft_filtered = img_fft * kernel_fft

    # Inverse Fourier Transform
    img_filtered = np.abs(ifft2(img_fft_filtered))

    # Convert to uint8 for image display
    return img_filtered



def gaussian_filter(image, kernel_size, sigma):
    """Lọc Gaussian cho ảnh."""
    #def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    gaussian_kernel_value  =  kernel / np.sum(kernel)

    filtered_image =  convolve2d(image, gaussian_kernel_value, mode='same', boundary="symm")
    return filtered_image


def median_filter(image, size):
    rows, cols = image.shape
    final_image = np.zeros(image.shape)

    for i in range(rows):
        for j in range(cols):
            neighborhood = image[max(0, i - size//2):min(rows, i + size//2 + 1),
                                max(0, j - size//2):min(cols, j + size//2 + 1)]

            # Apply median filter
            final_image[i, j] = np.median(neighborhood)

    return final_image