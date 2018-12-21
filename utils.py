import numpy as np
import cv2 as cv
import os
image_count = 0
def detect_hand(image, thre_area = 500, hmax = 200, hmin = 80, \
                smax = 200, smin = 100, vmax = 145, vmin = 10):
    '''
        args:
            image : 输入的待检测图像
        return:
            图像中是否含有手
            True : 检测到手
    '''
    temp_image = np.zeros(image.shape)
    bin_image = np.zeros(image.shape)
    temp_image = image.copy()
    hsv_image = cv.cvtColor(temp_image, cv.COLOR_BGR2HSV)
    height = hsv_image.shape[0]
    width = hsv_image.shape[1]
    channels = hsv_image.shape[2]
    area_count = 0
    for i in range(height):
        for j in range(width):
            H = hsv_image[i,j,0]
            S = hsv_image[i,j,1]
            V = hsv_image[i,j,2]
            if (H<hmax and H>hmin) and (S<smax and S>smin) \
              and (V<vmax and V>vmin):
              area_count += 1
              bin_image[i,j,:] = 255
    global image_count
    image_count+=1
    # DEBUG
    if not os.path.exists("output/debug"):
        os.makedirs("output/debug")
    # cv.imwrite("output/debug/image_{:05d}.jpg".format(image_count), temp_image)
    # np.save("output/debug/{:05d}.npy".format(image_count), hsv_image)
    # 判断手部面积是否超过阈值
    if area_count > thre_area:
        return True # 含有手
    else:
        return False # 无

