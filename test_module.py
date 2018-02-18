import numpy as np 
import cv2
def test_Points2LineEqABC():
	import cyutility
	answer = (np.random.rand(100,5)-0.5)*200
	# print np.where((answer[:,0]==0)&(answer[:,1]==0))
	# answer = np.delete(answer, np.where((answer[:,0]==0)&(answer[:,1]==0)))
	lines = np.zeros((answer.shape[0],4), dtype=np.int)
	nonZero = np.where(answer[:,1] != 0)
	lines[nonZero, 0] = answer[nonZero, 3]
	lines[nonZero, 2] = answer[nonZero, 4]
	lines[nonZero, 1] = ((-answer[nonZero,2]-answer[nonZero,0]*answer[nonZero,3])/answer[nonZero,1]).astype(int)

	lines[nonZero,3] = ((-answer[nonZero,2]-answer[nonZero,0]*answer[nonZero,4])/answer[nonZero,1]).astype(int)
	
	result = np.zeros((len(lines),5), dtype=np.float)
	cyutility.PyfromPoints2LineEq(lines[:,:2], lines[:,2:], result, mode=0)

	fromAnswer = np.zeros((1000,1000), dtype=np.uint8)
	fromResult = np.zeros((1000,1000), dtype=np.uint8)

	for l in lines:
		cv2.line(fromAnswer, tuple(l[:2]+500), tuple(l[2:]+500), 255,1)

	for r in result:
		x1 = int(r[3])
		y1 = int((-r[2] - (r[0]*r[3])) / r[1])
		x2 = int(r[4])
		y2 = int((-r[2] - (r[0]*r[4])) / r[1])
		cv2.line(fromResult, (x1+500,y1+500), (x2+500,y2+500), 255,1)

	img = np.hstack((fromResult,fromAnswer))
	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def test_Points2LineEqMC():
	import cyutility
	answer = (np.random.rand(100,4)-0.5)*200
	# print np.where((answer[:,0]==0)&(answer[:,1]==0))
	# answer = np.delete(answer, np.where((answer[:,0]==0)&(answer[:,1]==0)))
	# print "answer", answer[0]
	lines = np.zeros((answer.shape[0],4), dtype=np.int)

	lines[:, 0] = answer[:, 2]
	lines[:, 2] = answer[:, 3]
	lines[:, 1] = (answer[:,1]+answer[:,0]*answer[:,2]).astype(int)

	lines[:, 3] = (answer[:,1]+answer[:,0]*answer[:,3]).astype(int)
	result = np.zeros((len(lines),4), dtype=np.float)
	cyutility.PyfromPoints2LineEq(lines[:,:2], lines[:,2:], result, mode=1)

	fromAnswer = np.zeros((1000,1000), dtype=np.uint8)
	fromResult = np.zeros((1000,1000), dtype=np.uint8)

	for l in lines:
		cv2.line(fromAnswer, tuple(l[:2]+500), tuple(l[2:]+500), 255,1)

	for r in result:
		x1 = int(r[2])
		y1 = int(r[1] + (r[0]*r[2]))
		x2 = int(r[3])
		y2 = int(r[1] + (r[0]*r[3]))
		cv2.line(fromResult, (x1+500,y1+500), (x2+500,y2+500), 255,1)

	img = np.hstack((fromResult,fromAnswer))
	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def test_GroupingLineMC():
	import cyutility
	# sample = [[0,15,0,400],
	# 		[0,12,100,300],
	# 		[2, 9,200,200],
	# 		[2,18,150,200]]
	sample = [[ 2.18181818e-02, 3.18109091e+02, 2.00000000e+01, 2.20e+02],
		[ 2.50000000e-02, 3.16666667e+02, 3.00000000e+02, 3.40000000e+02],
		[ 3.17182663e-01, 2.77028896e+02, 4.20000000e+02, 7.60000000e+02],
		[ 2.07430341e-02, 3.16983832e+02, 4.20000000e+02, 7.60000000e+02],
		[-8.55357143e-02, 2.98633333e+02, 4.20000000e+02, 7.00000000e+02]]
	sample = np.array(sample, dtype=np.float)
	result = cyutility.PyGroupingLineMC(sample, 15)
	fromResult = np.zeros((640,960), dtype=np.uint8)
	print "result",result
	for r in result:
		x1 = int(r[2])
		y1 = int(r[1] + (r[0]*r[2]))
		x2 = int(r[3])
		y2 = int(r[1] + (r[0]*r[3]))
		cv2.line(fromResult, (x1+0,y1+0), (x2+0,y2+0), 255,1)
	cv2.imshow("img", fromResult)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
if __name__ == '__main__':
	# test_Points2LineEqABC()
	# test_Points2LineEqMC()
	test_GroupingLineMC()