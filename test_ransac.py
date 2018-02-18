import ransacLib
import numpy as np
import cv2




def create_random_sample(n_sample, w, h, m, c, inlier, scaleX = None, scaleY = None):
	point = np.random.rand(n_sample,2) 
	point[:,0] *= w
	point[:,1] *= h

	point[:inlier,1] = m*point[:inlier,0] + c

	if scaleX is None:
		scaleX = w/70
		# scaleX = 1
	if scaleY is None:
		scaleY = h/70
		# scaleY = 1
	point[:inlier,0] += np.random.normal(0, scaleX, inlier)
	point[:inlier,1] += np.random.normal(0, scaleY, inlier)
	return point.astype(int)

def ransac(points, maxIter, T):
	maxVote = 0
	c = np.zeros(3, dtype=float)
	bestSupporter = None
	for i in range(maxIter):
		sample_point = points[ransacLib.randomRansacSample(2, len(point)-1, 0),:]
		ransacLib.fitLine2Point(sample_point, c)
		listVer = ransacLib.verfyPoints(c, points, T)
		suppoter = np.where(listVer>0)[0]
		if len(suppoter) > maxVote:
			maxVote = len(suppoter)
			bestSupporter = points[suppoter,:]
	return bestSupporter

img = np.zeros((480,640,3), dtype = np.uint8)

point = create_random_sample(50, img.shape[1], img.shape[0], 0.7, 0, 35)

for p in point:
	cv2.circle(img, (p[1],p[0]), 3, 255, -1)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
w_f = lambda x,y: ransacLib.lineVerify(x,y,5)
f_f = lambda x,y: ransacLib.fitLine2Point(x,y)

coeff = np.array([0.7,-1,0], dtype=float)

e1 = cv2.getTickCount()

inlier = ransacLib.ransac(point, 5, 30.0)
inlier = point[np.where(inlier>0)[0],:]
# inlier = ransac(point, 100, 30.0)


e2 = cv2.getTickCount()
print inlier
print "frame rate is", cv2.getTickFrequency()/float(e2-e1)
for i,p in enumerate(point):
	cv2.circle(img, (p[1],p[0]), 3, (0,0,255), -1)

for i,p in enumerate(inlier):
	cv2.circle(img, (p[1],p[0]), 3, (0,255,0), -1)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()