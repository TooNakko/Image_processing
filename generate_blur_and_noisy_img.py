import cv2
import matplotlib.pyplot as plt
import numpy as np

def add_gaussian_noise(img, psnr):
    mean = 0
    sigma = 1000 *  10 ** (-psnr / 20.0)

    gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1])) 
    noisy_image = np.zeros(img.shape)
    noisy_image = img + gaussian
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    return noisy_image

def blur_image(image, kernel_size):
    kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel_2d = np.outer(kernel, kernel.transpose())
    blurred_image = cv2.filter2D(image, -1, kernel_2d)
    return blurred_image

def simulate_blur_and_noise(original_image, psnr, blur_kernel_size):
    blurred_image = blur_image(original_image, kernel_size=blur_kernel_size)
    noisy_blurred_image = add_gaussian_noise(blurred_image, psnr)
    return noisy_blurred_image



