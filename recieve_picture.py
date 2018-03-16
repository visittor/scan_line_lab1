import cv2
import numpy as np 
import sys
import time

def recieve_video_cam(id):
	cap = cv2.VideoCapture(id)
	# cap.set(3,240)
	# cap.set(4,240)
	# cap.set(5,320) #(240, 320, 3)

	cap.set(3,480)
	cap.set(4,480)
	cap.set(5,640) #(480, 640, 3)

	try:
		while 1 == 1:
			ret,img = cap.read()
			yield img
	finally:
		cap.release()

def recieve_video_file(file_name, repeat = 1):
	cap = cv2.VideoCapture(file_name)
	try:
		while  1 == 1:
			ret, img = cap.read()
			if not ret :
				cap.set( 2, 0)
				ret,img = cap.read()
			if ret:
				yield img
	finally:
		cap.release()

if __name__ == '__main__':
	is_write = 0
	is_puase = 1
	id_ = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
	reciver = recieve_video_cam(id_)
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
	img_ = next(reciver)
	img = img_
	while 1 == 1:
		if is_puase:
			img = next(reciver)
			img_ = img.copy()
		else:
			img = img_.copy()
		cv2.imshow('img',img)
		if is_write:
			out.write(img)
		k = cv2.waitKey(1)
		if k&0XFF == 27:
			break
		elif k&0XFF == ord('s'):
			is_write = (is_write+1)%2
			a = 'writing...' if is_write == 1 else "puase..."
			print a
		elif k&0xFF == ord("c"):
			cv2.imwrite("chessboard/"+str(time.time())+".png",img)
		elif k&0xFF == ord("p"):
			is_puase = 0 if is_puase == 1 else 1

	reciver.close()
	out.release()
	cv2.destroyAllWindows()
