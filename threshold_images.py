import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    mask_sobel = np.zeros_like(scaled_sobel)
    mask_sobel[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])] = 1
    return mask_sobel

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # scaled_sobelxy = np.uint8(sobelxy/np.max(sobelxy) * 255)
    mask_sobel = np.zeros_like(gray)
    scaled_sobelxy = gray
    mask_sobel[(scaled_sobelxy>=mag_thresh[0]) & (scaled_sobelxy<=mag_thresh[1])] = 1
    
    return mask_sobel

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)    
    arctans = np.arctan2(abs_sobely, abs_sobelx)
    mask = np.zeros_like(arctans)
    mask[(arctans >= thresh[0]) & (arctans <= thresh[1])] = 1
    
    return mask


ksize = 3 # Choose a larger odd number to smooth gradient measurements



for i in range(8):
    image = mpimg.imread('warped_images/test' + str(i+1))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    s_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 2]
    l_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 0]

    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))


    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    gradx_s = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady_s = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary_s = mag_thresh(s_channel, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary_s = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(0.7, 1.3))


    gradxy = np.zeros_like(gradx)
    gradxy[((gradx == 1) | (grady == 1))] = 1


    mag_dir_binary = np.zeros_like(dir_binary)
    mag_dir_binary[((mag_binary == 1) & (dir_binary == 0))] = 1

    combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) | (dir_binary == 1))] = 1
    combined[((gradx == 1) | (gradx_s == 1) | (mag_binary_s == 1))] = 1

    mpimg.imsave('thresholded_images/test' + str(i+1), combined)

    f, axes = plt.subplots(2, 4, figsize=(20,10))

    mag_v = mag_thresh(luv[:,:,2], mag_thresh=(160, 255))

    mag_b = mag_thresh(lab[:,:,2], mag_thresh=(145, 190))

    mag_s = mag_thresh(hsv[:,:,1], mag_thresh=(65, 200))


    combined = np.zeros_like(dir_binary)
    combined[((mag_v==1)|(mag_b==1)) | ((mag_b==1) & (mag_s==1 ))] = 1

    white = mag_thresh(hsv[:,:,2], mag_thresh=(200, 255))

    axes[0, 0].set_title('la_B')
    axes[0, 0].imshow(lab[:,:,2], cmap='gray')

    axes[0, 1].set_title('S')
    axes[0, 1].imshow(hsv[:,:,1], cmap='gray')

    axes[0, 2].set_title('V')
    axes[0, 2].imshow(hsv[:,:,2], cmap='gray')

    axes[0, 3].set_title('Combined')
    axes[0, 3].imshow(combined, cmap='gray')


    axes[1, 0].set_title('mag B')
    axes[1, 0].imshow(mag_b, cmap='gray')

    axes[1, 1].set_title('mag S')
    axes[1, 1].imshow(mag_s, cmap='gray')

    axes[1, 2].set_title('MAG V')
    axes[1, 2].imshow(mag_v, cmap='gray')

    axes[1, 3].set_title('IMAGE')
    axes[1, 3].imshow(image, cmap='gray')

    plt.show()

