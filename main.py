import generate_blur_and_noisy_img as gen_img
import matplotlib.pyplot as plt
import cv2  
import recovering as rcv
import plotting
import MSE_and_psnr as Map
import time

original_image = cv2.cvtColor(cv2.imread('HQ.jpg'), cv2.COLOR_BGR2GRAY)

print("Input the psnr:")
psnr = int(input())
print("\nInput the blur kernel size:")
blur_kernel_size = int(input())

print("Generating noisy image---")
temp_time = time.time()
noisy_image = gen_img.simulate_blur_and_noise(original_image, psnr, blur_kernel_size)
print("Took {0:.2f} seconds.\n".format(time.time() - temp_time))

print("Processing box blur method---")
temp_time = time.time()
box_blur_recovered_image = rcv.box_blur(noisy_image,10)
print("Took {0:.2f} seconds.\n".format(time.time() - temp_time))

temp_time = time.time()
print("Processing weiner method---")
weiner_recovered_image = rcv.wiener_filter(noisy_image, 5, 5)
print("Took {0:.2f} seconds.\n".format(time.time() - temp_time))

temp_time = time.time()
print("Processing gaussian filter method---")
gaussian_filter_recovered_image = rcv.gaussian_filter(noisy_image, 5, 1.5)
print("Took {0:.2f} seconds.\n".format(time.time() - temp_time))

temp_time = time.time()
print("Processing median filter method---")
median_filter_recovered_image = rcv.median_filter(noisy_image, 4)
print("Took {0:.2f} seconds.".format(time.time() - temp_time))


print("""
    MSE value between the original image and image recovered by:
    1. Blur box method: {0:.2f}
    2. Weiner method: {1:.2f}
    3. Gaussian filter method: {2:.2f}
    4. Median filter method: {3:.2f}
      
    psnr value between the original image and image recovered by:
    1. Blur box method: {4:.2f}
    2. Weiner method: {5:.2f}
    3. Gaussian filter method: {6:.2f}
    4. Median filter method: {7:.2f}
      """.format(Map.mse(original_image, box_blur_recovered_image), Map.mse(original_image, weiner_recovered_image),
                 Map.mse(original_image, gaussian_filter_recovered_image), Map.mse(original_image, weiner_recovered_image),
                 Map.psnr(original_image, box_blur_recovered_image), Map.psnr(original_image, weiner_recovered_image),
                 Map.psnr(original_image, gaussian_filter_recovered_image), Map.psnr(original_image, median_filter_recovered_image)))

plotting.plotting(original_image, noisy_image, box_blur_recovered_image, weiner_recovered_image, gaussian_filter_recovered_image, median_filter_recovered_image)
