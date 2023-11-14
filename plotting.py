import matplotlib.pyplot as plt

def plotting(original_image, noisy_image, box_blur_recovered_image, weiner_recovered_image, gaussian_recovered_image,richardson_lucy_recovered_image):
    plt.figure(figsize=(16, 8))

    plt.subplot(2,3, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title('Original image')
    plt.axis('off')
    plt.subplot(2,3, 2)
  
    plt.imshow(noisy_image, cmap="gray")
    plt.title('Noisy image' )
    plt.axis('off')
   
    plt.subplot(2,3, 3) 
    plt.imshow(box_blur_recovered_image, cmap="gray")
    plt.title('Box blur filter recovered Image' )
    plt.axis('off') 
    plt.subplot(2,3, 4)
    
    plt.imshow(weiner_recovered_image, cmap="gray")
    plt.title('Weiner filter recovered Image' )
    plt.axis('off')
    
    plt.subplot(2,3, 5)
    plt.imshow(gaussian_recovered_image, cmap="gray")
    plt.title('Gaussian filter recovered Image' )
    plt.axis('off')
    
    plt.subplot(2,3, 6)
    plt.imshow(richardson_lucy_recovered_image, cmap="gray")
    plt.title('Richardson and Lucy filter recovered Image' )
    plt.axis('off')
    
    plt.show()