import find_pattern
import cv2
import numpy as np
from recieve_picture import recieve_video_file,recieve_video_cam
from scanline import ScanLine
from line_provider import Line_provider
from Goal_provider import Goal_provider
from cross_provider import Cross_provider
from ScanLandmark import ScanLandmark
import time
import configobj
from find_pattern import find_color_pattern_polygon,visualize_polygon_scanline,to_region_from_polygon

file_name = "sample_with_robot_1.avi"
file_name = "sample_with_robot_2.avi"
file_name = "sample_with_robot_3.avi"
# file_name = "sample_7_no_robot.avi"
file_name = "goal_robot_ball_2.avi"
file_name = "dowaina_chattarin.avi"
# file_name = "dowaina_chattarin2.avi"
file_name = "dowaina_chattarin3.avi"
# file_name = "dowaina_chattarin4.avi"
file_name = "RCAP_A.avi"
reciever = recieve_video_file(file_name)
# reciever = recieve_video_cam(1)


config = configobj.ConfigObj("colordef_rcap_hanuman1.ini") ##color config
kwarg = {'line_angleThreshold':0.3, 'is_do_horizontal':False}##argument for class
print config
scanlandmark = ScanLandmark(config, **kwarg)##Create class instance

is_stop = 1
is_write = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

grammar = 1.2
lookup = np.array( [ 255.0*((float(i)/255.0)**(grammar)) for i in range(256) ], dtype = np.uint8 )
while 1 == 1:
	if is_stop:
		img_cap = next(reciever)
	img = img_cap.copy()
	img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	# img_hsv[:,:,2] = lookup[img_hsv[:,:,2]]s2.5, 1, -30, 0)
	from_region = np.zeros(img.shape, dtype = np.uint8) + 0
	from_region2 = np.zeros((img.shape[1],img.shape[0],img.shape[2]), dtype = np.uint8)

	e1 = cv2.getTickCount()

###############################################################################

	boundary = scanlandmark.do_scan_image(img, cvt2hsv = True, horizon =0 )
	goal = scanlandmark.get_goals()
	# robot = scanlandmark.get_robots( "cyan", "magenta")
###############################################################################

	e2 = cv2.getTickCount()

	time_used = (e2 - e1)/ cv2.getTickFrequency()
	cv2.putText(from_region,str(1/time_used),(0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
	scanlandmark.polygon_scanline.visualize(img, scanlandmark.color_dict)
	scanlandmark.vertical_scan.visualize_scan_line(from_region)
	cv2.drawContours(from_region, [cv2.convexHull(np.array([boundary]))], 0, (0,0,0), -1)
	
	##how to use output from get_goals##
	for gb in goal:
		cv2.line(img, gb.goal_base, gb.top, (255,0,170), 3)
		cv2.circle(img, gb.goal_base, 10, (170,0,170), -1)
	# for r in robot["Ally"]:
	# 	cv2.circle(img, (r.footPos[1], r.footPos[0]), 10, (0,170,170), -1)
	# for r in robot["Opponent"]:
	# 	cv2.circle(img, (r.footPos[1], r.footPos[0]), 10, (0,170,170), -1)

	# scanlandmark.visualize_horizontal_region(from_region)
	scanlandmark.visualize_vertical_region(img)

	time_used = (e2 - e1)/ cv2.getTickFrequency()
	cv2.putText(from_region,str(1/time_used),(0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

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
