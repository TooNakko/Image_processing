import generate_blur_and_noisy_img as gen_img
import matplotlib.pyplot as plt
import cv2  
import recovering as rcv
import plotting
import MSE_and_psnr as Map
import time

image_path = 'conrs.jpg'  #input your image's path here

original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)


print("Input the sigma:")
sigma = float(input())


print("\nInput the blur kernel size:")
blur_kernel_size = int(input())


print("Generating noisy image with sigma = {0} and blur kernel size = {1} ---".format(sigma, blur_kernel_size))
temp_time = time.time()
noisy_image = gen_img.simulate_blur_and_noise(original_image, sigma, blur_kernel_size)
psnr_of_noised_img = Map.psnr(original_image, noisy_image)
print("Took {0:.2f} seconds.\n".format(time.time() - temp_time))

print("Processing box blur method---")
temp_time = time.time()
box_blur_recovered_image = rcv.box_blur(noisy_image,10)
print("Took {0:.2f} seconds.\n".format(time.time() - temp_time))

temp_time = time.time()
print("Processing weiner method---")
weiner_recovered_image = rcv.wiener_filter(noisy_image, 5)
print("Took {0:.2f} seconds.\n".format(time.time() - temp_time))

temp_time = time.time()
print("Processing gaussian filter method---")
gaussian_recovered_image = rcv.gaussian_filter(noisy_image, 5, 1.5)
print("Took {0:.2f} seconds.\n".format(time.time() - temp_time))

temp_time = time.time()
print("Processing Richardson and Lucy filter method---")
richardson_lucy_recovered_image = rcv.richardson_lucy_filter(noisy_image, blur_kernel_size, 4) #
print("Took {0:.2f} seconds.".format(time.time() - temp_time))


print("""
    With sigma = {0:.4f}, blur kernel size = {1}, we have 
              MSE = {2:.2f} and psnr = {7:.1f}
      
    MSE value between the original image and image recovered by:
    1. Blur box filter method: {3:.2f}
    2. Weiner filter method: {4:.2f}
    3. Gaussian filter method: {5:.2f}
    4. Richardson and Lucy filter method: {6:.2f}
      """.format(sigma, blur_kernel_size, Map.mse(original_image, noisy_image), 
                 Map.mse(original_image, box_blur_recovered_image), Map.mse(original_image, weiner_recovered_image),
                 Map.mse(original_image, gaussian_recovered_image), Map.mse(original_image, richardson_lucy_recovered_image), 
                 psnr_of_noised_img))

plotting.plotting(original_image, noisy_image, box_blur_recovered_image, weiner_recovered_image, gaussian_recovered_image, richardson_lucy_recovered_image)
