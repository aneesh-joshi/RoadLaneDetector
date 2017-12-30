import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


for i in range(8):
    image = mpimg.imread('warped_images/test' + str(i+1))


    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

    f, axes = plt.subplots(3, 3, figsize=(20, 10))

    axes[0, 0].imshow(hls[:,:,0])
    axes[0, 0].set_title('IMAGE')

    axes[0, 1].imshow(hls[:, :, 1])
    axes[0, 1].set_title('L')

    axes[0, 2].imshow(hls[:, :, 2])
    axes[0, 2].set_title('S')

    axes[1, 0].imshow(lab[:, :, 0])
    axes[1, 0].set_title('LAB_L')

    axes[1, 1].imshow(hls[:, :, 1])
    axes[1, 1].set_title('LAB_A')

    axes[1, 2].imshow(hls[:, :, 2])
    axes[1, 2].set_title('LAB_B')

    axes[2, 0].imshow(hls[:, :, 0])
    axes[2, 0].set_title('LUV_L')

    axes[2, 1].imshow(hls[:, :, 1])
    axes[2, 1].set_title('LUV_U')

    axes[2, 2].imshow(hls[:, :, 2])
    axes[2, 2].set_title('LUV_V')

    for i in range(3):
        for j in range(3):
            axes[i, j].axis('off')


    plt.show()