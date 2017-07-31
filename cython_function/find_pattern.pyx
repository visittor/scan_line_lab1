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
@cython.nonecheck(False)
cdef unsigned int color_classify_c(int val1, int val2, int val3,np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int n_color):
	cdef int i
	for i in range(0,n_color):
		if color[i][0][0]<=val1 and color[i][1][0]>=val1:
			if color[i][0][1]<=val2 and color[i][1][1]>=val2:
				if color[i][0][2] <= val3 and color[i][1][2] >= val3:
					return i
	return i+1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_INT_t,ndim=3] find_color_pattern_x_c(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int grid_dis, int step, int co):
	cdef int max_w = img.shape[1]
	cdef int max_h = img.shape[0]
	cdef int n_color = color.shape[0]
	cdef int x = 0,y = 0
	cdef unsigned int current_color = -1
	cdef unsigned int loop_counter = 0
	cdef int color_count = 0
	cdef np.ndarray[DTYPE_INT_t,ndim = 2] out = np.zeros([(max_h//step)*(max_w//grid_dis)+1,3],dtype = DTYPE_INT)

	# for x in range(1,max_w,grid_dis):
	while x < max_w:
		# for y in range(1,max_h,step):
		while y < max_h:
			current_color = color_classify_c(img[y,x][0], img[y,x][1], img[y,x][2],color,n_color)
			out[color_count][0] = y
			out[color_count][1] = x
			out[color_count][2] = current_color
			img[y,x] = [0,0,255]
			color_count += 1
			loop_counter += 1
			y += step + loop_counter//co
		loop_counter = 0
		y = 0
		x += grid_dis
	return out

def find_color_pattern_x(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim=3] color, int grid_dis = 50, int step = 1, int co = 1080):
	if color.shape[1] != 2:
		raise ValueError("Wrong format color array.Color array must have shape (n_shape,2,3)")
	elif color.shape[2] != 3:
		raise ValueError("must be 3 channels color")
	return find_color_pattern_x_c(img,color,grid_dis,step,co)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_INT_t,ndim=3] find_color_pattern_y_c(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int grid_dis, int step, int co):
	cdef int max_w = img.shape[1]
	cdef int max_h = img.shape[0]
	cdef int n_color = color.shape[0]
	cdef int x = 0,y = 0
	cdef int current_color = -1
	cdef unsigned int loop_counter = 0
	cdef int color_count = 0
	cdef np.ndarray[DTYPE_INT_t,ndim = 2] out = np.zeros([(max_h//grid_dis)*(max_w//step)+1,3],dtype = DTYPE_INT)
	
	while y < max_h:
		while x < max_w:
			current_color = color_classify_c(img[y,x][0], img[y,x][1], img[y,x][2],color,n_color)
			out[color_count][0] = y
			out[color_count][1] = x
			out[color_count][2] = current_color
			img[y,x] = [0,0,255]
			x += step
		x = 0
		loop_counter += 1
		y += grid_dis + loop_counter//co
	return out

def find_color_pattern_y(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int grid_dis = 50, int step = 1, int co =  1080):
	if color.shape[1] != 2:
		raise ValueError("Wrong format color array.Color array must have shape (n_shape,2,3)")
	elif color.shape[2] != 3:
		raise ValueError("must be 3 channels color")
	return find_color_pattern_y_c(img,color,grid_dis,step,co)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_INT_t, ndim = 2] to_region_c(np.ndarray[DTYPE_INT_t, ndim = 2] pattern, int axis):
	cdef int max_size = pattern.shape[0]
	cdef unsigned int loop_counter = 0
	cdef int counter = 0
	cdef int color = -1
	cdef int grid_line = 0
	cdef np.ndarray[DTYPE_INT_t, ndim = 2] temp = np.zeros([max_size,4], dtype = DTYPE_INT)

	temp[counter,0] = pattern[loop_counter,axis]
	temp[counter, 1] = pattern[loop_counter, (axis+1)%2]
	temp[counter, 3] = pattern[loop_counter, 2]
	color = pattern[loop_counter,2]
	while loop_counter < max_size:

		if pattern[loop_counter, axis] != temp[counter, 0]:
			temp[counter, 2] = pattern[loop_counter - 1, (axis+1)%2]
			counter += 1
			temp[counter, 0] = pattern[loop_counter, axis]
			temp[counter, 1] = pattern[loop_counter, (axis+1)%2]
			temp[counter, 3] = pattern[loop_counter, 2]
		elif color != pattern[loop_counter,2]:
			temp[counter, 2] = pattern[loop_counter, (axis+1)%2]
			counter += 1
			temp[counter, 0] = pattern[loop_counter,axis]
			temp[counter, 1] = pattern[loop_counter, (axis+1)%2]
			temp[counter, 3] = pattern[loop_counter, 2]
			color = pattern[loop_counter,2]

		loop_counter += 1

	cdef np.ndarray[DTYPE_INT_t, ndim = 2] out = np.zeros([counter, 4], dtype = DTYPE_INT)
	for i in range(counter):
		out[i] = temp[i]

	return out

def to_region(np.ndarray[DTYPE_INT_t, ndim = 2] pattern, int axis):

	if pattern.shape[1] != 3:
		raise ValueError("It not a pattern from a function find_color_pattern")
	if axis > 2:
		raise ValueError("Axis must not exceed 1")

	return to_region_c(pattern, axis)