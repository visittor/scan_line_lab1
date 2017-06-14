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

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef int color_classify_c(np.ndarray[DTYPE_UINT8_t,ndim = 1] val,np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int n_color):
	cdef int i
	for i in range(0,n_color):
		if color[i][0][0]<=val[0] and color[i][1][0]>=val[0]:
			if color[i][0][1]<=val[1] and color[i][1][1]>=val[1]:
				if color[i][0][2] <= val[2] and color[i][1][2] >= val[2]:
					return i
	return i+1

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef np.ndarray[DTYPE_INT_t,ndim=3] find_color_pattern_x_c(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int grid_dis, int step):
	cdef int max_w = img.shape[1]
	cdef int max_h = img.shape[0]
	cdef int n_color = color.shape[0]
	cdef int x,y
	cdef int current_color = -1
	cdef int pre_color = -1
	cdef np.ndarray[DTYPE_INT_t,ndim = 1] color_count = np.zeros(n_color+1,dtype = DTYPE_INT)
	cdef np.ndarray[DTYPE_INT_t,ndim = 3] out = np.zeros([n_color+1,(max_h//step)*(max_w//grid_dis)+1,2],dtype = DTYPE_INT)

	for x in range(0,max_w,grid_dis):
		current_color = -1
		pre_color = -1
		for y in range(0,max_h,step):
			current_color = color_classify_c(img[y,x],color,n_color)
			if current_color != pre_color and current_color>-1 and y != 0:
				if pre_color > -1:
					out[current_color][color_count[current_color]][0] = y
					out[current_color][color_count[current_color]][1] = x
					out[pre_color][color_count[pre_color]][0] = y
					out[pre_color][color_count[pre_color]][1] = x
					color_count[pre_color] += 1
					color_count[current_color] += 1
				pre_color = current_color
	return out

def find_color_pattern_x(np.ndarray[DTYPE_UINT8_t,ndim = 3] img,np.ndarray[DTYPE_UINT8_t,ndim=3] color,int grid_dis = 50,int step = 1):
	if color.shape[1] != 2:
		raise ValueError("Wrong format color array.Color array must have shape (n_shape,2,3)")
	elif color.shape[2] != 3:
		raise ValueError("must be 3 channels color")

	return find_color_pattern_x_c(img,color,grid_dis,step)

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef np.ndarray[DTYPE_INT_t,ndim=3] find_color_pattern_y_c(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int grid_dis, int step):
	cdef int max_w = img.shape[1]
	cdef int max_h = img.shape[0]
	cdef int n_color = color.shape[0]
	cdef int x,y
	cdef int current_color = -1
	cdef int pre_color = -1
	cdef np.ndarray[DTYPE_INT_t,ndim = 1] color_count = np.zeros(n_color+1,dtype = DTYPE_INT)
	cdef np.ndarray[DTYPE_INT_t,ndim = 3] out = np.zeros([n_color+1,(max_h//grid_dis)*(max_w//step)+1,2],dtype = DTYPE_INT)
	
	for y in range(0,max_h,grid_dis):
		current_color = -1
		pre_color = -1
		for x in range(0,max_w,step):
			current_color = color_classify_c(img[y,x],color,n_color)
			if current_color != pre_color and current_color != -1 and x != 0:
				if pre_color > -1:
					out[current_color][color_count[current_color]][0] = y
					out[current_color][color_count[current_color]][1] = x
					out[pre_color][color_count[pre_color]][0] = y
					out[pre_color][color_count[pre_color]][1] = x
					color_count[pre_color] += 1
					color_count[current_color] += 1
				pre_color = current_color
	return out

def find_color_pattern_y(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int grid_dis, int step):
	if color.shape[1] != 2:
		raise ValueError("Wrong format color array.Color array must have shape (n_shape,2,3)")
	elif color.shape[2] != 3:
		raise ValueError("must be 3 channels color")

	return find_color_pattern_y_c(img,color,grid_dis,step)
	














