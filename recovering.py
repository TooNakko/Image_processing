import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d


def box_blur(image, kernel_size):
    # Tạo kernel có giá trị là 1
    kernel_value = np.ones((kernel_size, kernel_size))
    
    # Chuẩn hóa kernel để tổng giá trị bằng 1 (đối với box blur)
    kernel_value_normalized = kernel_value/ np.sum(kernel_value)
    
    # Áp dụng convolution
    blurred_image = convolve2d(image, kernel_value_normalized, mode='same', boundary='symm')
    
    return blurred_image

def wiener_filter(img, kernel_size):
    K = .25
    # Normalize the kernel 
    normalized_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    # Pad the kernel to match the size of the image
    padded_kernel = np.pad(normalized_kernel, [(0, img.shape[0] - normalized_kernel.shape[0]), (0, img.shape[1] - normalized_kernel.shape[1])], 'constant')

    # Fourier Transform of the image and kernel
    img_fft = fft2(img)
    kernel_fft = fft2(padded_kernel)

    # Wiener filter formula
    kernel_fft = np.conj(kernel_fft) / (np.multiply(np.conj(kernel_fft),kernel_fft)+ K)
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

def low_pass_filter(image, filter_type, filter_size):
    if filter_type == 'gaussian':
        kernel = cv2.getGaussianKernel(filter_size, 0)
        low_pass_image = cv2.filter2D(image, -1, np.outer(kernel, kernel.transpose()))
    elif filter_type == 'box':
        kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size ** 2)
        low_pass_image = cv2.filter2D(image, -1, kernel)
    else:
        raise ValueError("Unsupported filter type")

    return low_pass_image