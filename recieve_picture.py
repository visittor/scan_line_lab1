import cv2
import numpy as np 

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
			# img = cv2.GaussianBlur(img,(5,5),0)
			yield img
	finally:
		cap.release()


if __name__ == '__main__':
	is_write = 0
	reciver = recieve_video_cam(1)
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
	while 1 == 1:
		img = next(reciver)
		# print img.shape
		cv2.imshow('img',img)
		# print img.shape
		if is_write:
			out.write(img)
		k = cv2.waitKey(1)
		if k&0XFF == ord('q'):
			break
		elif k&0XFF == ord('s'):
			is_write = (is_write+1)%2
			a = 'writing...' if is_write == 1 else "puase..."
			print a

	out.release()
	cv2.destroyAllWindows()

