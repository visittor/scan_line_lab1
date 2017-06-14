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
cdef np.ndarray[DTYPE_INT_t,ndim = 4] angle_const_hist_c(np.ndarray[DTYPE_INT_t,ndim = 2] lines, int n_angle_bin, int n_const_bin, int min_angle, int max_angle, int min_const, int max_const):
	cdef int i,ang_bin,const_bin
	cdef int n_line = line.shape[0]
	cdef np.ndarray[DTYPE_INT_t,ndim = 4] out = np.zeros([2,n_angle_bin,n_const_bin,2], dtype = DTYPE_INT)

	for i in range(0,n_line):
		ang_bin = (lines[i][0] - min_angle)//(max_angle - min_angle)
		const_bin = (lines[i][1] - min_const)//(max_const - min_const)
		if const_bin > -1 and ang_bin > -1 and const_bin < 16 and ang_bin < 16:
			if out[1][ang_bin][const_bin][0] == 0:
				out[0][ang_bin][const_bin][0] = lines[i][0]
				out[0][ang_bin][const_bin][1] = lines[i][1]
			out[1][ang_bin][const_bin][0] += 1
	return out

def angle_const_hist(lines, n_angle_bin = 16, n_const_bin = 16, min_angle = -np.pi, max_angle = np.pi, min_const = 0, max_const = 750):
	if lines.ndim != 2:
		raise ValueError("lines array's ndim != 2")
	if lines.shape[1] != 2:
		raise ValueError("line.shape[1] != 2")

	return angle_const_hist_c(lines, n_angle_bin, n_const_bin, min_angle, max_angle, min_const, max_const)

