import numpy as np 

def unit_vector(vector):
	""" return unit_vector of vector"""
	return vector.astype(float)/np.linalg.norm(vector)

def angle_between(vec1, vec2):
	u_vec1 = unit_vector(vec1)
	u_vec2 = unit_vector(vec2)
	return np.arccos( np.clip(np.dot(u_vec1, u_vec2), -1, 1))

def check_(self,first,r):
		last_index = self._united_region[first][0]
		next_index = r
		if last_index != -1:
			v1 = np.array([self._region_output[first][0], (self._region_output[first][1] + self._region_output[first][2])/2 ]) - np.array([self._region_output[last_index][0], (self._region_output[last_index][1] + self._region_output[last_index][2])/2 ])
			v2 = np.array([self._region_output[next_index][0], (self._region_output[next_index][1] + self._region_output[next_index][2])/2 ]) - np.array([self._region_output[first][0], (self._region_output[first][1] + self._region_output[first][2])/2 ]) 
			angle = angle_between(v1, v2)
			if angle < 0.25 :
				return 0
			else:
				return 1
		return 0

def frechet_distance(m1, m2):
	max_dis = 0
	for i in m1:
		min_dis = -1
		for j in m2:
			if min_dis == -1:
				min_dis = np.linalg.norm(i - j)
			elif np.linalg.norm(i - j) < min_dis:
				min_dis = np.linalg.norm(i - j)
		if min_dis > max_dis :
			max_dis = min_dis
	return max_dis
