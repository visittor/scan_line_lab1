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

def find_line_intersection(m1, c1, m2, c2):
	a = np.array([ [m1, -1], [m2, -1] ])
	b = np.array([ -c1, -c2])
	return np.linalg.solve(a,b)

