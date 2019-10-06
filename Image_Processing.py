#!/usr/bin/Python3

import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2


file_dir = os.getcwd()
img_root_path = os.path.join(file_dir, "Images_Data")
img_path = os.path.join(img_root_path, 'lena.jpg')

def key_oper(key=27, timeout=0):
    k = cv2.waitKey(timeout)
    if k == key:
        cv2.destroyAllWindows()

# ============================ Image Read and Show ==============================
# Gray image
img_gray = cv2.imread(img_path, 0)
print(img_gray.shape)
cv2.imshow("lena-gray", img_gray)
key_oper()

# Color image
img = cv2.imread(img_path, 1)
print(img.shape)
cv2.imshow("lena-gray", img)
key_oper()

# Show image with matplotlib
B, G, R = cv2.split(img)
print(B.shape, G.shape, R.shape)
img_plt = cv2.merge((R, G, B))
plt.imshow(img_plt), plt.title("Lena")
plt.show()

# ============================ Image Basic Operation ==============================
# Image corp
img_corp = img[150:400, 150:400]
print(img_corp.shape)
cv2.imshow("lena-corp", img_corp)
key_oper()

# Change color
delta_R = 100
B, G, R = cv2.split(img)
R[R <= 155] += delta_R
R[R > 155] = 255
img_1 = cv2.merge((B, G, R))
cv2.imshow("lena-1", img_1)
key_oper()

def change_color(img):

    B, G, R = cv2.split(img)

    delta_b = random.randint(-100, 100)
    if delta_b > 0:
        blim = 255 - delta_b
        B[B > blim] = 255
        B[B <= blim] = (blim + B[B <= blim]).astype(img.dtype)
    elif delta_b < 0:
        blim = 0 - delta_b
        B[B < blim] = 0
        B[B >= blim] = (blim + B[B >= blim]).astype(img.dtype)

    delta_g = random.randint(-100, 100)
    if delta_g > 0:
        glim = 255 - delta_g
        G[G > glim] = 255
        G[G <= glim] = (glim + G[G <= glim]).astype(img.dtype)
    elif delta_g < 0:
        glim = 0 - delta_g
        G[G < glim] = 0
        G[G >= glim] = (glim + G[G >= glim]).astype(img.dtype)

    delta_r = random.randint(-100, 100)
    if delta_r > 0:
        rlim = 255 - delta_r
        R[R > rlim] = 255
        R[R <= rlim] = (rlim + R[R <= rlim]).astype(img.dtype)
    elif delta_r < 0:
        rlim = 0 - delta_r
        R[R < rlim] = 0
        R[R >= rlim] = (rlim + R[R >= rlim]).astype(img.dtype)

    img_new = cv2.merge((B, G, R))
    return img_new

img_2 = change_color(img)
cv2.imshow("lena-2", img_2)
key_oper()

# gamma correction
def adjust_gamma(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append((i / 255.0) ** inv_gamma * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img, table)

img_dark_path = os.path.join(img_root_path, "sudoku.png")
img_dark = cv2.imread(img_dark_path)
print(img_dark.shape)
cv2.imshow("sudoku", img_dark)
key_oper()
img_gct_1 = adjust_gamma(img_dark, gamma=0.7)
cv2.imshow("sudoku-1", img_gct_1)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
img_gct_2 = adjust_gamma(img_dark, gamma=2)
cv2.imshow("sudoku-2", img_gct_2)
key_oper()

# Histogram
plt.hist(img_gct_2.flatten(), 256, [0, 256], color='g'), plt.title("Histogram")
plt.show()

# YUV switch
img_gct_yuv = cv2.cvtColor(img_gct_2, code=cv2.COLOR_BGR2YUV)
cv2.imshow("sudoku-yuv", img_gct_yuv)
key_oper()
img_gct_yuv[:, :, 0] = cv2.equalizeHist(img_gct_yuv[:, :, 0])
img_gct_y = cv2.cvtColor(img_gct_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow("sudoku-2", img_gct_2)
cv2.imshow("sudoku-y", img_gct_y)
key_oper()

# ============================ Image Transform ==============================
# Rotation
rows, cols, chs = img.shape
Mr = cv2.getRotationMatrix2D((rows/2, cols/2), 60, 1)
print(Mr)
img_rot = cv2.warpAffine(img, Mr, (cols, rows))
cv2.imshow("lena-rotate", img_rot)
key_oper()

# Translation
Mt = np.float32([[1, 0, 50], [0, 1, 30]])
print(Mt)
img_trans = cv2.warpAffine(img, Mt, (cols, rows))
cv2.imshow("lena-trans", img_trans)
key_oper()

# Similarity Transform
Ms = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 0.5)
print(Ms)
img_sim = cv2.warpAffine(img, Ms, (cols, rows))
cv2.imshow("lena-sim", img_sim)
key_oper()

# Affine Transform
pts1 = np.float32([[50, 50], [cols-50, 50], [50, rows-50]])
pts2 = np.float32([[cols*0.2, rows*0.2], [cols*0.8, rows*0.1], [cols*0.2, rows*0.9]])
Ma = cv2.getAffineTransform(pts1, pts2)
print(Ma)
img_aff = cv2.warpAffine(img, Ma, (cols, rows))
cv2.imshow("lena-aff", img_aff)
key_oper()

# Perspective Transform
def perspective_trans(img, random_margin):
    height, width, channels = img.shape

    # src points
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    # dst points
    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    Mp = cv2.getPerspectiveTransform(pts1, pts2)
    img_per = cv2.warpPerspective(img, Mp, (width, height))
    return Mp, img_per

Mp, img_per = perspective_trans(img, 60)
print(Mp)
cv2.imshow("lena-per", img_per)
key_oper()

