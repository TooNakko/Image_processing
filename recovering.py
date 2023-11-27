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

def wiener_filter(img, psnr, kernel_size):
    
    K = .5
    # Normalize the kernel 
    normalized_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    # Pad the kernel to match the size of the image
    padded_kernel = np.pad(normalized_kernel, [(0, img.shape[0] - normalized_kernel.shape[0]), 
                                               (0, img.shape[1] - normalized_kernel.shape[1])],
                                                'constant')
    # Fourier Transform of the image and kernel
    img_fft = fft2(img)
    kernel_fft = fft2(padded_kernel)

    # Wiener filter formula
    kernel_fft = np.conj(kernel_fft) / (np.multiply(np.conj(kernel_fft),kernel_fft)+ K)
    img_fft_filtered = img_fft * kernel_fft

    # Inverse Fourier Transform
    final_image = np.abs(ifft2(img_fft_filtered))

    # Convert to uint8 for image display
    return final_image


def gaussian_filter(image, kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (kernel_size - 1) / 2) ** 2 +
                              (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    gaussian_kernel_value  =  kernel / np.sum(kernel)

    filtered_image =  convolve2d(image, gaussian_kernel_value,
                                  mode='same', boundary="symm")
    return filtered_image

def richardson_lucy_filter(img_noisy, kernel_size, iterations):
    kernel = cv2.getGaussianKernel(kernel_size, 0)
    psf = np.outer(kernel, kernel.transpose())
    psf /= psf.sum()
    # Define the initial estimate of the true image
    final_img = np.ones_like(img_noisy, dtype=float)
    # Iterate Richardson-Lucy deconvolution
    for _ in range(iterations):
        # Calculate the error term
        error = img_noisy / convolve2d(final_img, psf, 'same', 'symm')

        # Update the estimated image
        final_img *= convolve2d(error, np.flip(psf), 'same', 'symm')

    # Clip values to the valid range
    final_img = np.clip(final_img, 0, 255)

    # Convert to uint8 for image display
    final_img = np.uint8(final_img)

    return final_img