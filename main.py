import generate_blur_and_noisy_img as gen_img
import matplotlib.pyplot as plt
import cv2  
import recovering as rcv
import plotting
import MSE_and_psnr as Map
import time

image_path = 'HQ.jpg'

original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

while(True):
  print("Input the psnr:")
  psnr = int(input())
  if(psnr<=0):
    print("psnr must > 0")
    continue
  break

while(True):
  print("\nInput the blur kernel size:")
  blur_kernel_size = int(input())
  if(blur_kernel_size<=0):
    print("blur kernel size must > 0")
    continue
  break

print("Generating noisy image with psnr = {0} and blur kernel size = {1} ---".format(psnr, blur_kernel_size))
temp_time = time.time()
noisy_image = gen_img.simulate_blur_and_noise(original_image, psnr, blur_kernel_size)
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
richardson_lucy_recovered_image = rcv.richardson_lucy_filter(noisy_image, blur_kernel_size, 4)
print("Took {0:.2f} seconds.".format(time.time() - temp_time))


print("""
    With psnr = {0}, blur kernel size = {1}
    MSE value between the original image and image recovered by:
    1. Blur box filter method: {2:.2f}
    2. Weiner filter method: {3:.2f}
    3. Gaussian filter method: {4:.2f}
    4. Richardson and Lucy filter method: {5:.2f}
      """.format(psnr, blur_kernel_size, Map.mse(original_image, box_blur_recovered_image), Map.mse(original_image, weiner_recovered_image),
                 Map.mse(original_image, gaussian_recovered_image), Map.mse(original_image, richardson_lucy_recovered_image)))

plotting.plotting(original_image, noisy_image, box_blur_recovered_image, weiner_recovered_image, gaussian_recovered_image, richardson_lucy_recovered_image)
