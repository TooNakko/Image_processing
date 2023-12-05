import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import numpy as np
import cv2

#def add_gaussian_noise(img, psnr):
#    mean = 0
#    sigma = 255 * 10 ** (-psnr / 20.0)
#    gaussian_noise = np.random.normal(mean, sigma, img.shape)
#    noisy_image = img + gaussian_noise
#    noisy_image = np.clip(noisy_image, 0, 255)
#    return noisy_image

def add_gaussian_noise(img, sigma):
    mean = 0
    w = np.random.normal(mean, float(sigma), img.shape)
    noisy_image = img + w
    noisy_image = np.clip(noisy_image, 0, 255).astype(img.dtype)
    return noisy_image


def add_blur(image, kernel_size):
    kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel_2d = np.outer(kernel, kernel.transpose())
    blurred_image = cv2.filter2D(image, -1, kernel_2d)
    return blurred_image

def simulate_blur_and_noise(original_image, psnr, blur_kernel_size):
    blurred_image = add_blur(original_image, kernel_size=blur_kernel_size)
    noisy_blurred_image = add_gaussian_noise(blurred_image, psnr)
    return noisy_blurred_image



