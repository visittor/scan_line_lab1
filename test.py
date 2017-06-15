import find_pattern
import numpy as np 
import cv2
import time
import math

def test_find_color_pattern_y(img_name):
	img = cv2.imread(img_name)
	color_dict = {"green":([0,127,0],[127,255,127]),"white":([128,128,128],[255,255,255])}
	color_list = np.zeros((2,2,3),dtype = np.uint8)
	color_list[0,0] = np.array(color_dict["green"][0] , dtype = np.uint8)
	color_list[0,1] = np.array(color_dict["green"][1] , dtype = np.uint8) 
	color_list[1,0] = np.array(color_dict["white"][0] , dtype = np.uint8)
	color_list[1,1] = np.array(color_dict["white"][1] , dtype = np.uint8)
	print color_list[0][1]
	start = time.time()
	for i in range(0,20):
		out = find_pattern.find_color_pattern_y(img, color_list,50,2)
	stop = time.time()
	for i in out[1]:
		if (i == [0,0]).all():
			break
		cv2.circle(img,(i[1],i[0]),1,(255,0,0),-1)

	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow("image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print "---------- find_color_pattern_y ----------\n","time for function = ",(stop-start)/20,"frame rate = ",20/(stop-start),"\n------------------------------------------\n"
	return out

def test_find_color_pattern_x(img_name):
	img = cv2.imread(img_name)
	color_dict = {"green":([0,127,0],[127,255,127]),"white":([128,128,128],[255,255,255])}
	color_list = np.zeros((2,2,3),dtype = np.uint8)
	color_list[0,0] = np.array(color_dict["green"][0] , dtype = np.uint8)
	color_list[0,1] = np.array(color_dict["green"][1] , dtype = np.uint8) 
	color_list[1,0] = np.array(color_dict["white"][0] , dtype = np.uint8)
	color_list[1,1] = np.array(color_dict["white"][1] , dtype = np.uint8)
	print color_list[0][1]
	start = time.time()
	for i in range(0,20):
		out = find_pattern.find_color_pattern_x(img, color_list,50,2)
	stop = time.time()
	for i in out[1]:
		if (i == [0,0]).all():
			break
		cv2.circle(img,(i[1],i[0]),1,(255,0,0),-1)

	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow("image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print "---------- find_color_pattern_y ----------\n","time for function = ",(stop-start)/20,"frame rate = ",20/(stop-start),"\n------------------------------------------\n"
	return out

def test_find_grad(img_name,point):
	img = cv2.imread(img_name)
	img = cv2.pyrDown(img)
	img = cv2.pyrUp(img)
	times = []
	grads = []
	for i in point:
		if (i == [0,0]).all():
			break
		start = time.time()
		grad = find_pattern.find_grad(img,i[1],i[0],squre_size = 1)
		stop = time.time()
		times.append(float(stop-start))
		grads.append(grad)
	# print times
	# print "---------- find_grade ----------\n","time for function = ",sum(map(float,times))/float(len(times)), "frame rate = ",float(len(times))/sum(map(float,times)),"\n--------------------------------\n"
	return grads

def show_pic_test(img_name): # string
	img = cv2.imread(img_name)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow("image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def create_line(co,grads,img_name,lenght = 20):
	img = cv2.imread(img_name)
	ls_new_pos = []
	new_pos = []

	for i in range(len(grads)):
		x = co[1][i][1] + int( lenght * np.cos(grads[i]))
		y = co[1][i][0] + int( lenght * np.sin(grads[i]))

		new_pos.append(int(y))
		new_pos.append(int(x))
		ls_new_pos.append(new_pos)
		new_pos = []
		cv2.line(img,(co[1][i][1],co[1][i][0]),(int(x),int(y)),(255,0,0),1)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow("image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return ls_new_pos

if __name__ == "__main__":

	out = test_find_color_pattern_y("line_1.jpg")
	print out[1]
	grads = test_find_grad("line_1.jpg",out[1])

	d = create_line(out,grads,"line_1.jpg")

	# for i in range(len(d)):
	# 	print d[i]
