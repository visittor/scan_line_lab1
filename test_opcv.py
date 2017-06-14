import numpy as np
import cv2
import time 
from find_pattern import *
# Load an color image in grayscale
img = cv2.imread('1.jpg')

print img
print img.shape
print img.size

color_dic = {"green":([0,0,127],[127,127,255]),"white":([127,127,127],[255,255,255])}
color_arr = np.zeros([2,2,3],dtype = np.uint8)

i = 0
for k,v in color_dic.items():
	for j in range(0,3):
		color_arr[i][0][j] = v[0][j]
		color_arr[i][1][j] = v[1][j]
	i += 1
start = time.time()
for i in range(0,20):
	out = find_color_pattern_x(img,color_arr)
stop = time.time()
print "fps:",20/(stop-start)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()