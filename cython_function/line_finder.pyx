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

cdef np.ndarray[DTYPE_INT_t, ndim = 2] find_line_from_region_center_c(np.ndarray[DTYPE_INT_t, ndim = 2] region_cen, int axis):
	cdef int len_region = region_cen.shape[0]
	cdef int i, j, k, l
	cdef np.ndarray[DTYPE_INT_t, ndim = 2] linklist = np.zeros([len_region, 2], dtype = DTYPE_INT)
	cdef int counter = 0
	for i in range(0, len_region):
		linklist[i][0] -= 1
		linklist[i][1] -= 1
	i = 0
	j = 1
	k = 0
	while i < len_region - 1:
		while i+j < len_region and region_cen[i][axis] == region_cen[i+j][axis]:
			j += 1
		counter = j
		j = 1
		k = counter+i
		while k+j < len_region and region_cen[k][axis] == region_cen[k+j][axis]:
			j += 1
		if j == counter:
			for l in range(0, counter):
				linklist[i+l][1] = k+l
				linklist[k+l][0] = i+l
		i = k
		j = 1
	return linklist

cdef np.ndarray[DTYPE_INT_t, ndim = 2] line_from_linklist_c( np.ndarray[DTYPE_INT_t, ndim = 2] linklist, np.ndarray[DTYPE_INT_t, ndim = 2] region_cen):
	cdef int len_region = region_cen.shape[0]
	cdef np.ndarray[DTYPE_INT_t, ndim = 2] line_list_temp = np.zeros([len_region, 4], dtype = DTYPE_INT)
	cdef int index = 0
	cdef int counter = 0
	cdef int next_index
	while index < len_region:
		if linklist[index][0] == -1 and linklist[index][1] != -1:
			next_index = linklist[index][1]
			while linklist[next_index][1] != -1:
				next_index = linklist[next_index][1]
			line_list_temp[counter][0] = region_cen[index][0]
			line_list_temp[counter][1] = region_cen[index][1]
			line_list_temp[counter][2] = region_cen[next_index][0]
			line_list_temp[counter][3] = region_cen[next_index][1]
			counter += 1
		index += 1
	cdef np.ndarray[DTYPE_INT_t, ndim = 2] line_list = np.zeros([counter, 4], dtype = DTYPE_INT)
	for i in range(0, counter):
		line_list[i][0] = line_list_temp[i][0]
		line_list[i][1] = line_list_temp[i][1]
		line_list[i][2] = line_list_temp[i][2]
		line_list[i][3] = line_list_temp[i][3]

	return line_list

def find_line_from_region_center(region_cen, axis = 0):
	if region_cen.ndim != 2:
		raise ValueError("input array must ndim equal to 2")
	elif region_cen.shape[1] < axis-1 :
		raise ValueError("region.shape[1] < axis-1")
	cdef np.ndarray[DTYPE_INT_t, ndim = 2] link_list
	link_list = find_line_from_region_center_c( region_cen, axis)
	return line_from_linklist_c( link_list, region_cen)





