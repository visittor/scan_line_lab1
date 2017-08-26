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
	is_puase = 0
	reciver = recieve_video_cam(1)
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

		ret,thr = cv2.threshold(img[:,:,2],2127,255,cv2.THRESH_BINARY)
		_, contours, hierarchy  = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			for i in range(0,len(cnt),10):
				p = cnt[i]
				x,y = p[0]
				cv2.circle(img , (x,y), 1, (0,255,0), -1)
				cv2.putText(img,str(x)+","+str(y),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
		pixelpoints = np.transpose(np.nonzero(thr))
		# print np.amax(pixelpoints,axis = 0), np.amin(pixelpoints, axis = 0)
		cv2.imshow('img',img)
		cv2.imshow('thr',thr)
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
			cv2.imwrite("save__.jpg",img)
		elif k&0xFF == ord("p"):
			is_puase = 0 if is_puase == 1 else 1

	reciver.close()
	out.release()
	cv2.destroyAllWindows()

