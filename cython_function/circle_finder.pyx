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

cdef np.ndarray[DTYPE_INT_t, ndim = 3] find_circle_c(np.ndarray[DTYPE_INT_t, ndim = 2] line_hist, np.ndarray[DTYPE_INT_t, ndim = 2] indices, np.ndarray[DTYPE_INT_t, ndim = 2] points, np.ndarray[DTYPE_FLOAT_t, ndim = 2] lines, int w, int h, int max_r, int nbin_w, int nbin_h, int nbin_r):
	cdef int i,j,k,x,y
	cdef float r
	cdef nbin_ang = line_hist.shape[0]
	cdef nbin_const = line_hist.shape[1]
	cdef np.ndarray[DTYPE_INT_t, ndim = 3] out = np.zeros((nbin_h,nbin_w,nbin_r), dtype = DTYPE_INT)
	cdef np.ndarray[DTYPE_INT_t, ndim = 1] p1 = np.zeros([2], dtype = DTYPE_INT)
	cdef np.ndarray[DTYPE_INT_t, ndim = 1] p2 = np.zeros([2], dtype = DTYPE_INT)

	for i in range(0,nbin_ang//2):
		for j in range(0,nbin_const):
			if line_hist[i,j] != 0:
				for k in range(0, nbin_const):
					if line_hist[i+nbin_ang//2,k] != 0:
						p1 = points[indices[i,j]]
						p2 = points[indices[i+nbin_ang//2,k]]
						r = np.linalg.norm(p2 - p1)
						y = (p1[0]+p2[0])//2
						x = (p1[1]+p2[1])//2
						out[(nbin_h*y)//h, (nbin_w*x)//w, (nbin_r*r)//max_r] += 1
	return out

def find_circle(line_hist, indices, points, lines, w, h, max_r, nbin_w = 16, nbin_h = 16, nbin_r = 16):
	if line_hist.ndim != 2:
		raise ValueError("line_hist ndim must equal to 2")
	if indices.ndim != 2:
		raise ValueError("indices ndim must equal to 2")
	if points.ndim != 2:
		raise ValueError("points ndim must equal to 2")

	return find_circle_c(line_hist, indices, points, lines, w, h, max_r, nbin_w, nbin_h, nbin_r)	