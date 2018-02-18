import cv2
import numpy as np

file_name = "leipzig_012.avi"
cap = cv2.VideoCapture(file_name)
# cap = cv2.VideoCapture(1)
cv2.namedWindow('set')
def nothing(somethings):
	pass
cv2.createTrackbar('B1', 'set', 0, 255,nothing)
cv2.createTrackbar('B2', 'set', 0, 255, nothing)
cv2.createTrackbar('G1', 'set', 0, 255, nothing)
cv2.createTrackbar('G2', 'set', 0, 255, nothing)
cv2.createTrackbar('R1', 'set', 0, 255, nothing)
cv2.createTrackbar('R2', 'set', 0, 255, nothing)

is_puase = 0
# img = cv2.imread("pass_4.jpg")
# img = cv2.resize(img, None, fx = 0.25, fy = 0.25)
while True:
	if not is_puase:
		ret,frame = cap.read()
		print frame
		if not ret:
		    cap.set( 2, 0)
		    ret,frame = cap.read()
		# frame = img.copy()
	# frame = cv2.flip(frame,0)
		# frame = cv2.GaussianBlur(frame,(3,3),0)
	tran_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	H1 = cv2.getTrackbarPos('B1', 'set')
	H2 = cv2.getTrackbarPos('B2', 'set')
	S1 = cv2.getTrackbarPos('G1', 'set')
	S2 = cv2.getTrackbarPos('G2', 'set')
	V1 = cv2.getTrackbarPos('R1', 'set')
	V2 = cv2.getTrackbarPos('R2', 'set')
	lower = np.array([H1,S1,V1])
	upper = np.array([H2,S2,V2])
	mask = cv2.inRange(tran_frame,lower,upper)
	roi = cv2.bitwise_and(frame,frame,mask = mask)
	# circle = cv2.HoughCircles(frame[:, :, 2], cv2.HOUGH_GRADIENT, 1, 20, param1=85, param2=40, minRadius=0,
	#                           maxRadius=500)
	# if circle is not None:
	#     print 'a'
	#     for i in circle[0, :]:
	#         cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 0), 2)
	stack1 = np.hstack([frame, roi])
	stack2 = np.hstack([tran_frame[:, :, 0], tran_frame[:, :, 1], tran_frame[:, :, 2]])
	cv2.imshow('stack1',stack1)
	cv2.imshow('stack2',stack2)
	# stack = np.hstack((frame, mask))
	# cv2.imshow('window',stack)
	k = cv2.waitKey(10)
	if k == 27:
		break
	elif k == ord('p'):
		print lower, upper
	elif k == ord('s'):
		is_puase = (is_puase+1)%2
		a = "puase..." if is_puase else "resume..."
		print a
	elif k == ord('c'):
		cv2.imwrite("save.jpg", frame)

cv2.destroyAllWindows()
