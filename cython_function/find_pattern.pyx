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
		if color[i, 0, 0]<=val1 and color[i, 1, 0]>=val1:
			if color[i, 0, 1]<=val2 and color[i, 1, 1]>=val2:
				if color[i, 0, 2] <= val3 and color[i, 1, 2] >= val3:
					return i
	return i+1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_INT_t,ndim=2] find_color_pattern_x_c(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int grid_dis, int step, int co, int horizon, int end_scan):
	cdef int max_w = img.shape[1]
	# cdef int max_h = img.shape[0]
	cdef int max_h = end_scan
	cdef int n_color = color.shape[0]
	cdef unsigned int x = 0,y = horizon, y_ 
	cdef int current_color = -1, previous_color = -1
	cdef unsigned int loop_counter = 0
	cdef unsigned int color_count = 0
	cdef np.ndarray[DTYPE_INT_t,ndim = 2] temp = np.zeros([(((max_h-y)//step)+1)*((max_w//grid_dis)+1)+1,3],dtype = DTYPE_INT)
	if co == 0:
		co = 1
	while x < max_w:
		while y < max_h:
			current_color = color_classify_c(img[y,x,0], img[y,x,1], img[y,x,2],color,n_color)
			y_ = y
			if previous_color != current_color:
				previous_color = current_color
				while current_color == previous_color and y_>horizon:
					previous_color = color_classify_c(img[y_,x,0], img[y_,x,1], img[y_,x,2],color,n_color)
					y_ -= 1
			temp[color_count, 0] = y_
			temp[color_count, 1] = x
			temp[color_count, 2] = current_color
			previous_color = current_color
			color_count += 1
			loop_counter += 1
			y += step + loop_counter//co
		loop_counter = 0
		y = horizon
		x += grid_dis
	out = temp[:color_count,:]
	return out

def find_color_pattern_x(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim=3] color, int grid_dis = 50, int step = 1, int co = 1080, horizon = 0, end_scan = -1):
	if color.shape[1] != 2:
		raise ValueError("Wrong format color array.Color array must have shape (n_shape,2,3)")
	elif color.shape[2] != 3:
		raise ValueError("must be 3 channels color")
	if end_scan == -1:
		end_scan = img.shape[0]
	return find_color_pattern_x_c(img,color,grid_dis,step,co,horizon,end_scan)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_INT_t,ndim=2] find_color_pattern_y_c(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int grid_dis, int step, int co, int horizon, int end_scan):
	cdef int max_w = img.shape[1]
	cdef int max_h = end_scan
	cdef int n_color = color.shape[0]
	cdef int x = 0,y = horizon,x_
	cdef int current_color = -1, previous_color = -1
	cdef unsigned int loop_counter = 0
	cdef int color_count = 0
	cdef np.ndarray[DTYPE_INT_t,ndim = 2] temp = np.zeros([(((max_h-horizon)//grid_dis)+1)*((max_w//step)+1)+1,3],dtype = DTYPE_INT)
	while y < max_h:
		while x < max_w:
			current_color = color_classify_c(img[y,x,0], img[y,x,1], img[y,x,2],color,n_color)
			previous_color = current_color
			x_ = x
			if previous_color != current_color:
				previous_color = current_color
				while current_color == previous_color and x_>0:
					previous_color = color_classify_c(img[y,x_,0], img[y,x_,1], img[y,x_,2],color,n_color)
					x_ -= 1
			temp[color_count, 0] = y
			temp[color_count, 1] = x_
			temp[color_count, 2] = current_color
			previous_color = current_color
			color_count += 1
			x += step
		x = 0
		loop_counter += 1
		y += grid_dis
		# step = step - co if step > co else 1
		# step = step
	out = temp[:color_count,:]
	return out

def find_color_pattern_y(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int grid_dis = 50, int step = 1, int co =  1080, horizon = 0, end_scan = -1):
	if color.shape[1] != 2:
		raise ValueError("Wrong format color array.Color array must have shape (n_shape,2,3)")
	elif color.shape[2] != 3:
		raise ValueError("must be 3 channels color")
	if end_scan == -1:
		end_scan = img.shape[0]
	return find_color_pattern_y_c(img,color,grid_dis,step,co,horizon,end_scan)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_INT_t, ndim = 2] to_region_c(np.ndarray[DTYPE_INT_t, ndim = 2] pattern, int axis, int min_pixel):
	cdef int max_size = pattern.shape[0]
	cdef np.ndarray[DTYPE_INT_t, ndim = 2] temp = np.zeros([max_size,4], dtype = DTYPE_INT)
	cdef int loop_counter = 0
	cdef int prev_column = 0
	cdef int counter = 0
	cdef int color = -1
	cdef int grid_line = 0
	cdef int i
	temp[counter,0] = pattern[loop_counter,axis]
	temp[counter, 1] = pattern[loop_counter, (axis+1)%2]
	temp[counter, 3] = pattern[loop_counter, 2]
	color = pattern[loop_counter,2]
	prev_column = pattern[loop_counter,axis]
	loop_counter += 1
	while loop_counter < max_size:
		if pattern[loop_counter, axis] != temp[counter, 0]:
			temp[counter, 2] = pattern[loop_counter - 1, (axis+1)%2]
			counter += 1
			temp[counter, 0] = pattern[loop_counter, axis]
			temp[counter, 1] = pattern[loop_counter, (axis+1)%2]
			temp[counter, 3] = pattern[loop_counter, 2]
			color = pattern[loop_counter,2]
		elif color != pattern[loop_counter,2]:
			if pattern[loop_counter, (axis+1)%2] - temp[counter, 1] < min_pixel:
				if prev_column != temp[counter,0]:
					temp[counter, 0] = pattern[loop_counter, axis]
					temp[counter, 1] = pattern[loop_counter, (axis+1)%2]
					temp[counter, 3] = pattern[loop_counter, 2]
					color = pattern[loop_counter,2]
				elif counter > 0:
					counter -= 1
					color = temp[counter, 3]
				else:
					counter = 0
					color = temp[counter, 3]
			else:
				temp[counter, 2] = pattern[loop_counter, (axis+1)%2]
				counter += 1
				prev_column = pattern[loop_counter,axis]
				temp[counter, 0] = pattern[loop_counter,axis]
				temp[counter, 1] = pattern[loop_counter, (axis+1)%2]
				temp[counter, 3] = pattern[loop_counter, 2]
				color = pattern[loop_counter,2]

		loop_counter += 1
	temp[counter, 2] = pattern[loop_counter-1, (axis+1)%2]
	out = temp[:counter+1,:]

	return out

def to_region(np.ndarray[DTYPE_INT_t, ndim = 2] pattern, int axis, min_pixel = 1):

	if pattern.shape[1] != 3:
		raise ValueError("It not a pattern from a function find_color_pattern")
	if axis > 2:
		raise ValueError("Axis must not exceed 1")

	return to_region_c(pattern, axis, min_pixel)

cdef np.ndarray[DTYPE_INT_t, ndim = 1] find_color_circular_pattern_c(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int x, int y, int true_color, int square_size,int step):
	cdef int n_color = color.shape[0] 
	cdef int i,j,counter = 0,perimeter = 0
	cdef int max_w = img.shape[1]
	cdef int max_h = img.shape[0]
	i = x - square_size//2 if x > square_size//2 else 0
	j = y - square_size//2 if y > square_size//2 else 0
	perimeter += (x - i) + 1 + (y - j) + 1
	perimeter += square_size//2 if max_w - x > square_size//2 else max_w - x
	perimeter += square_size//2 if max_h - y > square_size//2 else max_h - y
	perimeter = (perimeter*2) - 4
	cdef np.ndarray[DTYPE_INT_t, ndim = 1] out = np.zeros([perimeter], dtype = DTYPE_INT)

	while i < x + square_size//2 and i < max_w - 1:
		if color_classify_c(img[j,i][0], img[j,i][1], img[j,i][2],color,n_color) == true_color:
			out[counter] = 1
		else:
			out[counter] = 0
		counter += 1
		i += step

	while j < y + square_size//2 and j < max_h - 1:
		if color_classify_c(img[j,i][0], img[j,i][1], img[j,i][2],color,n_color) == true_color:
			out[counter] = 1
		else:
			out[counter] = 0
		counter += 1
		j += step

	while i > x - square_size//2 and i > 0:
		if color_classify_c(img[j,i,0], img[j,i,1], img[j,i,2],color,n_color) == true_color:
			out[counter] = 1
		else:
			out[counter] = 0
		counter += 1
		i -= step

	while j > y - square_size//2 and j > 0:
		if color_classify_c(img[j,i][0], img[j,i][1], img[j,i][2],color,n_color) == true_color:
			out[counter] = 1
		else:
			print "perimeter :", perimeter, "counter :", counter
			out[counter] = 0
		counter += 1
		j -= step

	return out

def find_color_circular_pattern(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, int x, int y, int true_color, int square_size = 25, int step = 1):
	if color.shape[1] != 2:
		raise ValueError("Wrong format color array.Color array must have shape (n_shape,2,3)")
	elif color.shape[2] != 3:
		raise ValueError("must be 3 channels color")
	elif true_color < 0:
		raise ValueError("true_color's value must be more than 0")

	return find_color_circular_pattern_c(img, color, x, y, true_color, square_size, step)

cdef np.ndarray[DTYPE_INT_t,ndim=2] find_color_pattern_polygon_c(np.ndarray[DTYPE_UINT8_t,ndim = 3] img, np.ndarray[DTYPE_UINT8_t,ndim = 3] color, np.ndarray[DTYPE_INT_t,ndim=2] polygon,int step):
	cdef int max_h = img.shape[0]
	cdef int max_w = img.shape[1]
	cdef int polygon_lenght = polygon.shape[0]
	cdef np.ndarray[DTYPE_INT_t,ndim = 2] temp = np.zeros([(max_h*max_w)+1,3],dtype = DTYPE_INT)
	cdef int x1,y1,x2,y2,x,y
	cdef int diffX, diffY, maxStep, i, count = 0
	cdef int n_color = color.shape[0]
	cdef int current_color = -1
	cdef int index = 0

	while index < polygon_lenght - 1:
		x1 = polygon[index,0]
		y1 = polygon[index,1]
		x2 = polygon[index+1,0]
		y2 = polygon[index+1,1]
		diffX = x2-x1
		diffY = y2-y1
		maxStep = np.absolute(diffX) if np.absolute(diffX)>np.absolute(diffY) else np.absolute(diffY)
		i = 0
		while i < maxStep:
			x = x1 + ((diffX*i)//maxStep)
			x = x if x<max_w else max_w -1
			y = y1 + ((diffY*i)//maxStep)
			y = y if y < max_h else max_h - 1
			current_color = color_classify_c(img[y,x,0], img[y,x,1], img[y,x,2],color,n_color)
			temp[count, 0] = y
			temp[count, 1] = x
			temp[count, 2] = current_color
			count += 1
			i += step
		index += 1

	out = temp[:count,:]
	return out

def find_color_pattern_polygon( img, color, polygon,step = 5):
	return find_color_pattern_polygon_c(img, color, polygon, step)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_INT_t, ndim = 2] to_region_from_polygon_c(np.ndarray[DTYPE_INT_t, ndim = 2] pattern, int min_pixel):
	cdef int max_size = pattern.shape[0]
	cdef np.ndarray[DTYPE_INT_t, ndim = 2] temp = np.zeros([max_size,5], dtype = DTYPE_INT)
	cdef unsigned int loop_counter = 0, previous_counter = 0, preprevious_counter = 0
	cdef int counter = 0
	cdef int color = -1
	cdef int grid_line = 0
	cdef int i
	temp[counter,0] = pattern[loop_counter,0]
	temp[counter, 1] = pattern[loop_counter, 1]
	temp[counter, 4] = pattern[loop_counter, 2]
	color = pattern[loop_counter,2]
	loop_counter += 1
	while loop_counter < max_size:
		if color != pattern[loop_counter,2]:
			if loop_counter - previous_counter < min_pixel:
				previous_counter = preprevious_counter
				if counter > 0:
					counter -= 1
				else:
					counter = 0
				color = temp[counter, 4]
			else:
				preprevious_counter = previous_counter
				previous_counter = loop_counter
				temp[counter, 2] = pattern[loop_counter, 0]
				temp[counter, 3] = pattern[loop_counter, 1]
				counter += 1
				temp[counter, 0] = pattern[loop_counter,0]
				temp[counter, 1] = pattern[loop_counter, 1]
				temp[counter, 4] = pattern[loop_counter, 2]
				color = pattern[loop_counter,2]
		loop_counter += 1
	out = temp[:counter,:]
	return out

def to_region_from_polygon(pattern, minpix = 1):
	return to_region_from_polygon_c(pattern, minpix)

cpdef visualize_polygon_scanline(np.ndarray[DTYPE_UINT8_t,ndim=3] img, np.ndarray[DTYPE_INT_t,ndim=2] scanline, color_dict):
	cdef int i = 0
	cdef int scanlineLenght = scanline.shape[0]
	cdef int color_index
	while i < scanlineLenght:
		color_index = scanline[i,2]
		# print color_index
		for color in color_dict.values():
			if color_index == color.index:
				img[scanline[i,0],scanline[i,1],0] = np.uint8(color.RenderColor_RGB[0])
				img[scanline[i,0],scanline[i,1],1] = np.uint8(color.RenderColor_RGB[1])
				img[scanline[i,0],scanline[i,1],2] = np.uint8(color.RenderColor_RGB[2])
		i += 1
