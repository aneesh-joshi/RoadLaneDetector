import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_img(img):
    plt.imshow(img)
    plt.show()  

calibration_images = []

for filename in glob.iglob('camera_cal/calibration*.jpg', recursive=True):
    calibration_images.append(mpimg.imread(filename))

obj_points = [] # all coordinates
img_points = [] # coordinates of corners

objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for img in calibration_images:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret == True:

        img_points.append(corners)
        obj_points.append(objp)

        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[1::-1], None, None)

import pickle

for i, filename in enumerate(glob.iglob('camera_cal/calibration*.jpg', recursive=True)):
    img = mpimg.imread(filename)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('ORIGINAL')
    ax1.imshow(img)

    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    ax2.imshow(undistorted_img)
    ax2.set_title('UNDISTORTED')

    # mpimg.imsave('undistorted_images/test' + str(i+1), undistorted_img)
    plt.show()

with open('undistortion_data.pkl', 'wb') as f:
    pickle.dump([ret, mtx, dist, rvecs, tvecs], f)

print('UNDISTORTING IMAGES')

for i, filename in enumerate(glob.iglob('test_images/*.jpg', recursive=True)):
    img = mpimg.imread(filename)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('ORIGINAL')
    ax1.imshow(img)

    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    ax2.imshow(undistorted_img)
    ax2.set_title('UNDISTORTED')

    mpimg.imsave('undistorted_images/test' + str(i+1), undistorted_img)
    # plt.show()
