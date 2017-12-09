import numpy as np
import cv2
from util import *
from region_reciever import Region_reciver
from scanline import ScanLine

class Goal_provider( Region_reciver ):
	def __init__(self, angle_threshold, size_ratio, h2w_ratio = 2.0, distance_thr = 75):
		super( Goal_provider, self).__init__()
		super( Goal_provider, self).set_size_ratio(size_ratio)
		super( Goal_provider, self).set_angle_threshold(angle_threshold)
		self._region = None
		self._united_region = None
		self._h2w_ratio = h2w_ratio
		self._distance_thr = distance_thr

	def receive_region(self, region, connected_color = 0):
		self._region = region
		self._united_region = super( Goal_provider, self).unite_region(self._region, connected_color)

	def visualize_united_region(self, img, axis = 1):
		super( Goal_provider, self).visualize_united_region(img, self._united_region, axis)

	def LinkList2Squar(self):
		point_array = []
		for i in self._united_region:
			temp = np.zeros((2,2), dtype = np.int)
			if i.is_head() and i.is_alone() == 0:
				temp[0] = [i.region.column, i.region.start]
				i_ = i.forward
				while i_.is_tail() == 0:
					i_ = i_.forward
				temp[1] = [i_.region.column, i_.region.stop]
				point_array.append(temp)
		return point_array

	def filter(self, point_array, boundary):
		filtered = []
		for p1, p2 in point_array:
			h, w = p1 - p2
			if float(w) != 0 and float(h)/float(w) > self._h2w_ratio:
				if boundary is not None:
					distance = cv2.pointPolygonTest(boundary, (p2[1], p2[0]), True)
					# print distance
					if  - distance < self._distance_thr:
						filtered.append(np.array([p1,p2]))
				else: 
					filtered.append(np.array([p1,p2]))
		return filtered

	def get_filtred_Squar(self, boundary = None):
		return self.filter(self.LinkList2Squar(), boundary)

	@property
	def h2w_ratio(self):
		return self._h2w_retio

	def set_h2w_ratio(self, value):
		self._h2w_ratio = float(value)

	@property
	def distance_thr(self):
		return self._distance_thr

	def set_distance_thr(self, value):
		self._distance_thr = value