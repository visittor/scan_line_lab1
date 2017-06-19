import find_pattern
import line_finder
import filter_and_analyze
import circle_finder
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
		out = find_pattern.find_color_pattern_x(img, color_list,25,2)
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
	img = cv2.pyrDown(img)
	img = cv2.pyrDown(img)
	img = cv2.pyrUp(img)
	img = cv2.pyrUp(img)
	img = cv2.pyrUp(img)
	times = []
	grads = []
	for i in point:
		if (i == [0,0]).all():
			break
		start = time.time()
		grad = line_finder.find_grad(img,i[1],i[0],squre_size = 1)
		stop = time.time()
		times.append(float(stop-start))
		grads.append(grad)
	# print times
	# print "---------- find_grade ----------\n","time for function = ",sum(map(float,times))/float(len(times)), "frame rate = ",float(len(times))/sum(map(float,times)),"\n--------------------------------\n"
	return grads

def test_linear_eq(img_name,point):
	img = cv2.imread(img_name)
	img = cv2.pyrDown(img)
	img = cv2.pyrDown(img)
	img = cv2.pyrDown(img)
	img = cv2.pyrUp(img)
	img = cv2.pyrUp(img)
	img = cv2.pyrUp(img)
	start = time.time()
	for i in range(0,20):
		lines = line_finder.linear_eq(img,point,100,1)
	stop = time.time()
	print lines
	print "---------- linear_eq ----------\n","time for function = ",(stop-start)/20,"frame rate = ",20/(stop-start),"\n------------------------------------------\n"
	return lines

def test_angle_const_hist(img_name,lines_equa):
	img = cv2.imread(img_name)
	w,h,_ = img.shape
	start = time.time()
	for i in range(0,300):
		hist = filter_and_analyze.angle_const_hist(lines_equa,  n_const_bin = 16, max_const = np.sqrt(w*w + h*h))
	stop = time.time()
	print "---------- angle_const_hist ----------\n","time for function = ",(stop-start)/300,"frame rate = ",300/(stop-start),"\n------------------------------------------\n" 
	stop = time.time()
	return hist

def test_finde_circle_line(img_name, line_hist, indices, points, lines):
	img = cv2.imread(img_name)
	h,w,_ = img.shape
	print img.shape

	start = time.time()
	for i in range(0,300):
		circle_hist = circle_finder.find_circle(line_hist, indices, points, lines, w, h, w, nbin_w = 32, nbin_h = 32, nbin_r = 16)
	stop = time.time()
	
	print "---------- finde_circle_line ----------\n","time for function = ",(stop-start)/300,"frame rate = ",300/(stop-start),"\n------------------------------------------\n" 
	return circle_hist

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

def create_line_from_eq(equa, img_name):
	img = cv2.imread(img_name)

	for i in equa:
		coor1 = np.nan_to_num(np.array([i[1]/np.cos(i[0]),0]))
		coor2 = np.nan_to_num(np.array([(i[1]+np.sin(i[0])*700)/np.cos(i[0]), 700]))
		cv2.line(img,(int(coor1[1]),int(coor1[0])),(int(coor2[1]),int(coor2[0])),(255,0,0),1)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow("image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def create_line_from_hist(img_name, lines, hist, points):
	img = cv2.imread(img_name)
	line_indices,bin_ = hist[0],hist[1]
	real_line = np.where( bin_ > 5 )
	circle_line = np.where( bin_ < 5 )
	for i in line_indices[real_line]:
		line = lines[i]
		coor1 = np.nan_to_num(np.array([line[1]/np.cos(line[0]),0]))
		coor2 = np.nan_to_num(np.array([(line[1]+np.sin(line[0])*700)/np.cos(line[0]), 700]))
		cv2.line(img,(int(coor1[1]),int(coor1[0])),(int(coor2[1]),int(coor2[0])),(0,0,255),2)
	for i,j in zip(real_line[0],real_line[1]):
		bin_[i][j] = 0
	h,w,_ = img.shape
	print img.shape
	blank = np.zeros((h/10,w/10,1),dtype = np.uint8)
	for i in line_indices[circle_line]:
		point = points[i]/10
		# print point
		blank[point[0],point[1]] = 255
	# point = points/10
	# blank[point[:,0],point[:,1]] = 255
	circles = cv2.HoughCircles(blank,cv2.HOUGH_GRADIENT,1,15,param1=25,param2=10,minRadius=2,maxRadius=80)
	print circles
	try:
		for i in circles[0,:]:
		    # draw the outer circle
		    cv2.circle(img,(int(i[0]*10),int(i[1]*10)),int(i[2]*10),(0,0,0),2)
		    # draw the center of the circle
		    cv2.circle(img,(int(i[0]*10),int(i[1]*10)),2,(0,0,0),3) 
	except TypeError as e:
		print "TypeError", e

	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.namedWindow('blank', cv2.WINDOW_NORMAL)
	cv2.imshow("image",img)
	cv2.imshow("blank",blank)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def create_circle_from_hist(image_name, circle_hist):
	img = cv2.imread(image_name)
	h,w,_ = img.shape
	indices = np.where( circle_hist > 4 )
	for i,j,k in zip(indices[0], indices[1], indices[2]):
		cv2.circle(img, (w*j/32,h*i/32),w*k/32,(255,0,0),2)
	print indices
	print circle_hist.max()
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow("image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	image_name = "circle_2.jpg"
	out = test_find_color_pattern_x(image_name)
	print out[1]
	grads = test_find_grad(image_name,out[1])

	d = create_line(out,grads,image_name)

	lines = test_linear_eq(image_name,out[1])
	create_line_from_eq(lines, image_name)

	hist = test_angle_const_hist(image_name,lines)
	print hist
	create_line_from_hist(image_name, lines, hist, out[1])

	circle_hist = test_finde_circle_line(image_name, hist[1], hist[0], out[1], lines)
	print circle_hist
	create_circle_from_hist(image_name, circle_hist)
	# for i in range(len(d)):
	# 	print d[i]
