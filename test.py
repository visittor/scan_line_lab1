import find_pattern
import cv2
import numpy as np
from recieve_picture import recieve_video_file
from scanline import ScanLine
from line_provider import Line_provider
from Goal_provider import Goal_provider
from cross_provider import Cross_provider
from ScanLandmark import ScanLandmark
import time
import configobj

file_name = "sample_with_robot_1.avi"
file_name = "sample_with_robot_2.avi"
# file_name = "sample_with_robot_3.avi"
# file_name = "sample_7_no_robot.avi"
# file_name = "goal_robot_ball_2.avi"
reciever = recieve_video_file(file_name)

config = configobj.ConfigObj("colordef.ini")

kwarg = {''}

color_dict = {"black":( [ 0, 0, 0], [ 255, 45, 120 ]), "white":( [ 0, 0, 120], [ 255, 45, 255 ]), "green":( [ 30, 45, 0 ], [ 100, 255, 255 ]), }
color_list = np.zeros((3,2,3),dtype = np.uint8)
color_list[0,0] = np.array(color_dict["white"][0] , dtype = np.uint8)
color_list[0,1] = np.array(color_dict["white"][1] , dtype = np.uint8) 
color_list[1,0] = np.array(color_dict["green"][0] , dtype = np.uint8)
color_list[1,1] = np.array(color_dict["green"][1] , dtype = np.uint8)
color_list[2,0] = np.array(color_dict["black"][0] , dtype = np.uint8)
color_list[2,1] = np.array(color_dict["black"][1] , dtype = np.uint8)

sc = ScanLine(color_list = color_list, grid_dis = 20, scan_axis = 1, co = 2, step = 1)
sc2 = ScanLine(color_list = color_list, grid_dis = 20, scan_axis = 0, co = 1, step = 5)
# lp = Line_provider(angle_threshold = 1.047, size_ratio=0.5)
lp = Line_provider(angle_threshold = 0.4, size_ratio=0.5)
cp = Cross_provider()
gp = Goal_provider( 0.4, 0.3)

is_stop = 1
is_write = 0

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,480))

grammar = 1.1
lookup = np.array( [ 255.0*((float(i)/255.0)**(grammar)) for i in range(256) ], dtype = np.uint8 )
while 1 == 1:
	if is_stop:
		img_cap = next(reciever)
	img = img_cap.copy()
	img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	img_hsv[:,:,2] = lookup[img_hsv[:,:,2]]
	from_region = np.zeros(img.shape, dtype = np.uint8)
	from_region2 = np.zeros((img.shape[1],img.shape[0],img.shape[2]), dtype = np.uint8)

	e1 = cv2.getTickCount()

	sc.find_region(img_hsv, horizon = 0, end_scan = -1)
	boundary = sc.clip_region(1)
	end_scan = max(boundary, key = lambda x : x[1])[1]

	boundary.append(np.array([img.shape[1],img.shape[0]]))
	boundary.append(np.array([0,img.shape[0]]))
	cv2.drawContours(img_hsv, [cv2.convexHull(np.array([boundary]))], 0, (0,0,0), -1)

	sc2.find_region(img_hsv, horizon = 0, end_scan = end_scan)

	lp.receive_region(sc)
	lp.make_line(axis = 1, frechet_d_thr = 20)
	gp.receive_region(sc2)
	square = gp.get_filtred_Squar(boundary = cv2.convexHull(np.array([boundary])))

	cp.receive_line(lp)
	cross_point = cp.get_point()

	e2 = cv2.getTickCount()
	time_used = (e2 - e1)/ cv2.getTickFrequency()
	cv2.putText(from_region,str(1/time_used),(0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

	# for p in cross_point:
	# 	img_ = img[p[1]-50:p[1]+50, p[0]-50:p[0]+50]
	# 	cv2.imwrite("cross_sample/"+str(time.time())+".png", img_)
	# 	time.sleep(0.01)

	for p1, p2 in square:
		cv2.rectangle(img, (p1[1], p1[0]), (p2[1], p2[0]), (255,0,0), 3)

	for line in lp:
		x1,y1 = line.start
		x2,y2 = line.stop
		color = (0,100,255) if line.color == 0 else (0,0,0)
		cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

	for p in cross_point:
		if p.type == p.L:
			cv2.circle(img, tuple(p.coordinate.astype(int)), 15, (0,25,255), -1)
			cv2.putText(img, "L", (int(tuple(p.coordinate.astype(int))[0]) - 9, int(tuple(p.coordinate.astype(int))[1]) + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
		elif p.type == p.T:
			cv2.circle(img, tuple(p.coordinate.astype(int)), 15, (0,25,255), -1)
			cv2.putText(img, "T", (int(tuple(p.coordinate.astype(int))[0]) - 9, int(tuple(p.coordinate.astype(int))[1]) + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
		elif p.type == p.X:
			cv2.circle(img, tuple(p.coordinate.astype(int)), 15, (0,25,255), -1)
			cv2.putText(img, "X", (int(tuple(p.coordinate.astype(int))[0]) - 9, int(tuple(p.coordinate.astype(int))[1]) + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
		else:
			cv2.circle(img, tuple(p.coordinate.astype(int)), 15, (0,25,255), -1)
			cv2.putText(img, "?", (int(tuple(p.coordinate.astype(int))[0]) - 9, int(tuple(p.coordinate.astype(int))[1]) + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)

	sc2.visualize_region(from_region)
	# sc.visualize_scan_line(from_region)
	sc.visualize_region(from_region)
	cv2.drawContours(img, [cv2.convexHull(np.array([boundary]))], 0, (0,0,255), 1)
	# lp.visualize_united_region(from_region, axis = 1)
	lp.visualize_node(from_region)
	# cp.visualize_cross(img, circle_size = 5)

	# gp.visualize_united_region(img, axis = 0)

	out_img = np.hstack([img, from_region])
	cv2.imshow("img", out_img)

	if is_write:
		out.write(out_img)

	k = cv2.waitKey(10)
	if k == 27:
		break
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