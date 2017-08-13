from line_provider import Line_provider
from util import *
import numpy as np 
import cv2


class Cross_provider(object):

	_lines = None
	_points = []

	def __init__(self):
		pass

	def recive_line(self, line_, max_h = 480, max_w = 640):
		if line_.__class__ == Line_provider:
			self._lines = line_
			self._points = []
			self.find_all_cross(max_h, max_w)
		else:
			raise ValueError("line_ is not class Line_provider.Line")

	def find_intersection_point(self, l1, l2):
		point = find_line_intersection(l1.m, l1.c, l2.m, l2.c)
		return point

	def find_all_cross(self, max_h, max_w):
		ii = 0
		while ii < self._lines.lenght:
			jj = ii + 1
			while jj < self._lines.lenght:
				point_ = self.find_intersection_point(self._lines[ii], self._lines[jj])
				if 0 <= point_[0] < max_w and 0 <= point_[1] < max_h:
					self._points.append(point_)
				jj += 1
			ii += 1

	def visualize_cross(self, img, circle_size = 1, color = [255, 0, 0]):
		for p in self._points:
			cv2.circle(img, (int(p[0]), int(p[1])), circle_size, color, -1)

	def get_points(self):
		return self._points