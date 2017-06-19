from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPE_INT = np.int 
ctypedef np.int_t DTYPE_INT_t

DTYPE_UINT8 = np.uint8
ctypedef np.uint8_t DTYPE_UINT8_t

DTYPE_FLOAT = np.float 
ctypedef np.float_t DTYPE_FLOAT_t

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef float find_grad_c(np.ndarray[DTYPE_UINT8_t,ndim = 3] img,int x,int y,int squre_size):
	cdef int max_h = img.shape[0]
	cdef int max_w = img.shape[1]
	cdef float dy,dx,ang
	cdef np.ndarray[DTYPE_UINT8_t,ndim = 1] y1 = np.zeros(3,dtype = DTYPE_UINT8)
	cdef np.ndarray[DTYPE_UINT8_t,ndim = 1] y2 = np.zeros(3,dtype = DTYPE_UINT8)
	cdef np.ndarray[DTYPE_UINT8_t,ndim = 1] x1 = np.zeros(3,dtype = DTYPE_UINT8)
	cdef np.ndarray[DTYPE_UINT8_t,ndim = 1] x2 = np.zeros(3,dtype = DTYPE_UINT8)
	cdef int i,j
	for i in range(-squre_size//2,(squre_size//2)+1):
		for j in range(-squre_size//2,(squre_size//2)+1):
			if -1 < y+1+i < max_h and -1 < x+1+j < max_w:
				y1[0] += img[y+1+i,x+1+j,0]
				y1[1] += img[y+1+i,x+1+j,1]
				y1[2] += img[y+1+i,x+1+j,2]
				if -1 < y+i and -1 < x+j:
					y2[0] += img[y+i,x+j,0]
					y2[2] += img[y+i,x+j,1]
					y2[1] += img[y+i,x+j,2] 
			if -1 < y+1+i < max_h and -1 < x+j < max_w:
				x1[0] += img[y+1+i,x+j,0]
				x1[1] += img[y+1+i,x+j,1]
				x1[2] += img[y+1+i,x+j,2]
				if -1 < y+i and x+1+j < max_w:
					x2[0] += img[y+i,x+1+j,0]
					x2[2] += img[y+i,x+1+j,1]
					x2[1] += img[y+i,x+1+j,2]
	y1[0] = y1[0]//squre_size*squre_size
	y1[1] = y1[1]//squre_size*squre_size
	y1[2] = y1[2]//squre_size*squre_size
	y2[0] = y2[0]//squre_size*squre_size
	y2[1] = y2[1]//squre_size*squre_size
	y2[2] = y2[2]//squre_size*squre_size
	x1[0] = x1[0]//squre_size*squre_size
	x1[1] = x1[1]//squre_size*squre_size
	x1[2] = x1[2]//squre_size*squre_size
	x2[0] = x2[0]//squre_size*squre_size
	x2[1] = x2[1]//squre_size*squre_size
	x2[2] = x2[2]//squre_size*squre_size
	dy = np.linalg.norm(y1) - np.linalg.norm(y2)
	dx = np.linalg.norm(x1) - np.linalg.norm(x2)
	ang = -np.arctan2(dy,dx) + np.pi/4# tangent = 90 - grad
	return ang

def find_grad(np.ndarray[DTYPE_UINT8_t,ndim = 3] img,int x,int y,int squre_size = 3):
	if img.shape[2] != 3:
		raise ValueError("Image must have 3 color channels")
	return find_grad_c(img,x,y,squre_size)


cdef np.ndarray[DTYPE_FLOAT_t,ndim = 2] linear_eq_c(np.ndarray[DTYPE_UINT8_t, ndim = 3] img, np.ndarray[DTYPE_INT_t, ndim = 2] region, int max_index, int squre_size):
	cdef int i 
	cdef np.ndarray[DTYPE_FLOAT_t, ndim = 2] out = np.zeros([max_index,2], dtype = DTYPE_FLOAT)
	for i in range(0,max_index):
		if region[i][0] == 0 and region[i][1] == 0:
			break
		grad = find_grad_c(img,region[i][1],region[i][0],squre_size)
		c = np.cos(grad)*region[i][0] - np.sin(grad)*region[i][1]    #sin(w)x - cos(w)y + c = 0
		out[i][0] = grad
		out[i][1] = c
	return out

def linear_eq(img, region, max_index = 100, squre_size = 1):

	return linear_eq_c(img,region,max_index,squre_size)











