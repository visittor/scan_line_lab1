import cv2
import numpy as np 
import time

def find_roi(img, interest_color, approx = -1):
	mask = cv2.inRange(img.copy(), interest_color[0], interest_color[1])
	kernel = np.ones((15,15),np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	kernel = np.ones((5,5),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 10)
	mask = cv2.erode(mask,kernel,iterations = 15)
	im2, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	A = cv2.contourArea(cnt)
	for i in contours[1:]:
		if cv2.contourArea(i) > A:
			cnt = i
	hull = cv2.convexHull(cnt ,returnPoints = True)

	if approx == -1:
		return cnt, hull, mask
	else:
		epsilon = approx*cv2.arcLength(cnt,True)
		cnt = cv2.approxPolyDP(cnt,epsilon,True)

	return cnt, hull, mask

if __name__ == '__main__':
	img = cv2.imread("sample_9.jpg")
	img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	
	color_dict = {"green":( [ 41, 25, 0 ], [ 77, 255, 255 ]),"white":( [ 57, 0, 0 ], [ 100, 255, 255 ])}
	# mask = cv2.inRange(img.copy(), color_dict["green"][0], color_dict["green"][1])
	contour, hull, im2 = find_roi(img_hsv, ( np.array(color_dict["green"][0]), np.array(color_dict["green"][1]) ), approx = 0.01)
	for i in contour:
		 cv2.circle(img,(i[0][0],i[0][1]),2,(0,0,0),-1) 
	cv2.drawContours(img, [hull], 0, (0,0,255), 1)

	print contour.shape
	print len(contour)

	cv2.imshow("img",img)
	cv2.imshow("im2",im2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()