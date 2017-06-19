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