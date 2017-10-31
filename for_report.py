import find_pattern
import cv2
import numpy as np
from recieve_picture import recieve_video_file
from scanline import ScanLine
from line_provider import Line_provider
from cross_provider import Cross_provider 

file_name = "sample_with_robot_1.avi"
reciever = recieve_video_file(file_name)

color_dict = {"white":( [ 0, 0, 100], [ 255, 41, 255 ]), "green":( [ 30, 41, 0 ], [ 100, 255, 255 ])}
color_list = np.zeros((2,2,3),dtype = np.uint8)
color_list[0,0] = np.array(color_dict["white"][0] , dtype = np.uint8)
color_list[0,1] = np.array(color_dict["white"][1] , dtype = np.uint8) 
color_list[1,0] = np.array(color_dict["green"][0] , dtype = np.uint8)
color_list[1,1] = np.array(color_dict["green"][1] , dtype = np.uint8)

sc = ScanLine(color_list = color_list, grid_dis = 20, scan_axis = 0, co = 1, step = 2)
lp = Line_provider()
cp = Cross_provider()

is_stop = 1
is_write = 0

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,480))

while 1 == 1:
	if is_stop:
		img = next(reciever)
		img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		
	from_scanline = np.zeros(img.shape, dtype = np.uint8)	
	from_region = np.zeros(img.shape, dtype = np.uint8)
	from_region_cliped = np.zeros(img.shape, dtype = np.uint8)

	line_region = np.zeros(img.shape, dtype = np.uint8)
	line_no_filter = img.copy()
	line_filtered = img.copy()

	e1 = cv2.getTickCount()
	sc.find_region(img_hsv, horizon = 50)
	# sc.visualize_region(from_region)
	# sc.clip_region(1)
	# lp.recive_region(sc)
	# lines_ = lp.get_lines()
	# line_nofilter = lp.to_line_eq()
	e2 = cv2.getTickCount()
	time = (e2 - e1)/ cv2.getTickFrequency()
	cv2.putText(from_region,str(1/time),(0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
	# for l in lines_:
	# 	y1 = int(l[0]*l[2] + l[1])
	# 	y2 = int(l[0]*l[3] + l[1])
	# 	color = (0,100,255) if l[4] == 0 else (0,0,0)
	# 	cv2.line(line_filtered, (int(l[2]), y1), (int(l[3]), y2), color, 3)

	# for l in line_nofilter:
	# 	y1 = int(l[0]*l[2] + l[1])
	# 	y2 = int(l[0]*l[3] + l[1])
	# 	color = (255,100,0) if l[4] == 0 else (0,0,0)
	# 	cv2.line(line_no_filter, (int(l[2]), y1), (int(l[3]), y2), color, 3)

	# sc.visualize_scan_line(from_region)
	sc.visualize_region(from_region)

	# lp.visualize_united_region(line_region)
	# cp.visualize_cross(img, circle_size = 5)

	out_img = np.hstack([ from_region, line_filtered])
	cv2.imshow("original", img)
	cv2.imshow("img", out_img)

	if is_write:
		out.write(out_img)

	k = cv2.waitKey(10)
	if k == 27:
		break
	elif k == ord('a'):
		cv2.imwrite("out_img.jpg",out_img)
		cv2.imwrite("img.jpg",img)
	elif k == ord('s'):
		is_stop  = (is_stop+1)%2
		a = 'puase' if is_stop == 0 else 'continue'
		print a
	elif k&0XFF == ord('c'):
		is_write = (is_write+1)%2
		a = 'writing...' if is_write == 1 else "stop writing..."
		print a

reciever.close()
out.release()
cv2.destroyAllWindows()