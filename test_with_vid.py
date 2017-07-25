from recieve_picture import recieve_video_file
import find_pattern
import line_finder
import filter_and_analyze
import circle_finder
import numpy as np 
import cv2
import time
import math
from region_of_interest import find_roi

file_name = "sample_3.avi"
reciever = recieve_video_file(file_name)

color_dict = {"green":( [ 36, 41, 0 ], [ 49, 255, 149 ]),"white":( [ 56, 0, 0], [ 100, 255, 255 ]), "green_for_roi":( [ 41, 25, 0 ], [ 77, 255, 255 ])}
color_list = np.zeros((2,2,3),dtype = np.uint8)
color_list[0,0] = np.array(color_dict["green"][0] , dtype = np.uint8)
color_list[0,1] = np.array(color_dict["green"][1] , dtype = np.uint8) 
color_list[1,0] = np.array(color_dict["white"][0] , dtype = np.uint8)
color_list[1,1] = np.array(color_dict["white"][1] , dtype = np.uint8)

is_stop = 0
hc = cv2.CascadeClassifier("ball_new_14s.xml")

def find_ball_haar(img):
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	footballs = hc.detectMultiScale(img_gray,1.1,8)
	return footballs


while 1 == 1:
	if is_stop == 0:
		img = next(reciever)
		img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		contour, hull, mask = find_roi(img_hsv, ( np.array(color_dict["green_for_roi"][0]), np.array(color_dict["green_for_roi"][1]) ), approx = 0.01)	
		cv2.drawContours(mask, [hull], 0, 255, -1)
		img_hsv = cv2.bitwise_and(img_hsv,img_hsv, mask = mask)
		img = cv2.bitwise_and(img,img, mask = mask)
		img_hsv = cv2.blur(img_hsv,(11,11))
		# img_gray = cv2.cvtColor(img,cv21.COLOR_BGR2GRAY)
	# footballs = hc.detectMultiScale(img_gray,1.1,8)
	footballs = find_ball_haar(img)
	for x,y,h,w in footballs:
		img_hsv[y:y+w, x:x+h] = [0, 0, 0]
		img[y:y+w, x:x+h] = [0, 0, 255]

	out = find_pattern.find_color_pattern_x(img_hsv, color_list,30,4)
	region_center = find_pattern.find_region_center(out[1], axis = 1)
	line_list = line_finder.find_line_from_region_center(region_center, axis = 1)

	for i in out[1]:
		cv2.circle(img, (i[1],i[0]), 1, (0,255,255), -1)
	for i in line_list:
		cv2.line(img,(i[1],i[0]),(i[3],i[2]),(255,0,255),2)
	for i in region_center:
		cv2.circle(img,(i[1],i[0]),2,(255,0,0),-1)

	cv2.imshow("img", img)
	k = cv2.waitKey(1)
	if k == 27:
		break
	elif k == ord('s'):
		is_stop = (is_stop+1)%2
		a = "pause..." if is_stop else "continue..."
		print a


cv2.destroyAllWindows()
reciever.close()