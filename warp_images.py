import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np


for i in range(8):
    image = mpimg.imread('undistorted_images/test' + str(i+1))

    

    src = np.float32([[  595,  450 ],
            [  685,  450 ],
            [ 1000,  660 ],
            [  280,  660 ]])

    dst = np.float32([[  300,    0 ],
            [  980,    0 ],
            [  980,  720 ],
            [  300,  720 ]])


    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR )

    f, axes = plt.subplots(1,2, figsize=(20, 10))

    for point in src:
        plt.plot(point[0], point[1], '.')

    

    axes[1].imshow(image)
    axes[0].imshow(warped)
    plt.show()
    mpimg.imsave('warped_images/test' + str(i+1), warped, cmap='gray')