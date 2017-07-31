import find_pattern
import cv2
import numpy as np
from recieve_picture import recieve_video_file
from scanline import ScanLine

file_name = "sample_3.avi"
reciever = recieve_video_file(file_name)

color_dict = {"white":( [ 0, 0, 70], [ 255, 70, 255 ]), "green":( [ 36, 70, 0 ], [ 70, 255, 149 ])}
color_list = np.zeros((2,2,3),dtype = np.uint8)
color_list[0,0] = np.array(color_dict["white"][0] , dtype = np.uint8)
color_list[0,1] = np.array(color_dict["white"][1] , dtype = np.uint8) 
color_list[1,0] = np.array(color_dict["green"][0] , dtype = np.uint8)
color_list[1,1] = np.array(color_dict["green"][1] , dtype = np.uint8)

sc = ScanLine(color_list = color_list)

is_stop = 1
is_write = 0

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,480))

while 1 == 1:
	if is_stop:
		# img = cv2.imread("sample_3.jpg")
		img = next(reciever)
		img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		
	from_region = np.zeros(img.shape, dtype = np.uint8)

	# out = find_pattern.find_color_pattern_x(img_hsv.copy(), color_list, grid_dis = 25, co = 5)
	# region = find_pattern.to_region(out, 1)
	# print region
	# for i in region:
	# 	if i[3] == 0:
	# 		# cv2.line(from_region,(i[0],i[1]),(i[0],i[2]),(255,255,255),10)
	# 		from_region[i[1]:i[2],i[0]-12:i[0]+12] = [255,255,255]
	# 	elif i[3] == 1:
	# 		# cv2.line(from_region,(i[0],i[1]),(i[0],i[2]),(0,255,0),10)
	# 		from_region[i[1]:i[2],i[0]-12:i[0]+12] = [0,255,0]
	# 	else:
	# 		# cv2.line(from_region,(i[0],i[1]),(i[0],i[2]),(0,0,0),10)
	# 		from_region[i[1]:i[2],i[0]-12:i[0]+12] = [100,100,100]
	# 	cv2.circle(from_region,(i[0],i[1]),2,(0,0,255),-1)
	# 	cv2.circle(from_region,(i[0],i[2]),2,(0,0,255),-1)
	sc.find_region(img_hsv)
	sc.clip_region(1)
	sc.unite_region()
	lines_ = sc.to_line_eq()
	# point_array = sc.link_list_to_list()
	# for p,c in point_array:
	# 	print p[0], p[-1]
	# 	color = (0,0,255) if c == 0 else (0,0,0)
	# 	cv2.line(img, (p[0][0], p[0][1]), (p[-1][0], p[-1][1]), color, 2)
	for l in lines_:
		y1 = int(l[0]*l[2] + l[1])
		y2 = int(l[0]*l[3] + l[1])
		color = (0,100,255) if l[4] == 0 else (0,0,0)
		cv2.line(img, (int(l[2]), y1), (int(l[3]), y2), color, 3)

	sc.visualize_region(from_region)
	sc.visualize_united_region(from_region)

	out_img = np.hstack([img, from_region])
	# print out_img.shape
	cv2.imshow("img", out_img)

	if is_write:
			out.write(out_img)

	k = cv2.waitKey(1)
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