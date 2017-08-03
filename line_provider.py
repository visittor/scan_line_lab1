import numpy as np
import cv2
from util import *

class Line_provider(object):
	_united_region = []
	_region = None
	lines = []

	def __init__(self):
		pass

	def recive_region(self, region):
		self._region = region
		self.unite_region()
		
	def check_(self, first, r):
		if first.is_head() == 0:
			v1 = first.region - first.backward.region
			v2 = r.region - first.region
			if angle_between(v1, v2) < 0.41:
				return 0
			else:
				return 1
		return 0

	def connect_region(self, first, r):
		if first is None:
			return None
		while first < self._region.lenght and self._region[first].stop < self._region[r].start and self._region[first].next_column == self._region[r].column:
			first += 1
		if self._region[first].next_column != self._region[r].column:
			return None
		if self._region[first].start <= self._region[r].stop and self._region[first].color == self._region[r].color:
			if not self.check_(self._united_region[first], self._united_region[r]) and self._united_region[first].is_tail():
				self._united_region[first].forward = self._united_region[r]
				self._united_region[r].backward = self._united_region[first]
		first_ = first + 1
		while first < self._region.lenght and self._region[first_].start <= self._region[r].stop:
			if self._region[first_].next_column != self._region[r].column:
				return first
			if self._region[first_].stop >= self._region[r].start and self._region[first_].color == self._region[r].color:
				if not  self.check_(self._united_region[first_], self._united_region[r]) and self._united_region[first_].is_tail():
					self._united_region[first_].forward = self._united_region[r]
					self._united_region[r].backward = self._united_region[first_]
			else:
				return first
			first_ += 1
		return first

	def unite_region(self):	
		self._united_region = [ self.linklist(self._region[i]) for i in range(self._region.lenght)]
		first = None
		last = None
		r_ = None
		for r in range(self._region.lenght):
			if self._region[r].color == 1:
				continue
			if r_ is None or self._region[r_].column != self._region[r].column:
				last = first
				first = r
			last = self.connect_region(last, r)
			r_ = r

	def visualize_united_region(self, img):
		for i in self._united_region:
			if i.is_tail() == 0:
				start = i.region
				stop = i.forward.region
				cv2.line(img,(start.column,start.start),(stop.column,stop.start),(0,0,255),2)
				cv2.line(img,(start.column,start.stop),(stop.column,stop.stop),(0,0,255),2)
				if i.is_head():
					cv2.line(img,(start.column,start.start),(start.column,start.stop),(0,0,255),2)
				if i.forward.is_tail():
					cv2.line(img,(stop.column,stop.start),(stop.column,stop.stop),(0,0,255),2)

	def link_list_to_list(self):
		point_array = []
		for ii in self._united_region:
			temp = np.zeros((0,2), dtype = np.int)
			if ii.is_head() and ii.is_alone() == 0:
				temp = np.append(temp, [[ii.region.column, ii.region.middle]], axis = 0)
				# temp.append([ii.region.column, ii.region.middle])
				ii_ = ii.forward
				while ii_.is_tail() == 0:
					temp = np.append(temp, [[ii_.region.column, ii_.region.middle]], axis = 0)
					ii_ = ii_.forward
				if temp.shape[0] > 1:
					point_array.append((temp, ii.region.color))
		return point_array

	def to_line_eq(self):
		point_array = self.link_list_to_list()
		lines_ = []
		for p,color in point_array:
			x = p[:,0]
			y = p[:,1]
			A = np.vstack([x, np.ones(len(x))]).T
			m, c = np.linalg.lstsq(A, y)[0]
			lines_.append([m,c,x[0],x[-1],color,p.shape[0]])
		return lines_

	def compare_lines(self, l1, l2):
		m1 = [np.array([ 0, l1[1] ]), np.array([100, 100*l1[0]+l1[1] ]) ]
		m2 = [np.array([ 0, l2[1] ]), np.array([100, 100*l2[0]+l2[1] ]) ]
		frechet_d = frechet_distance(m1, m2)
		if frechet_d < 15:
			return 1
		return 0 

	def merge_lines(self, l1, l2):
		l1[0] = (l1[5]*l1[0] + l2[5]*l2[0])/(l1[5]+l2[5])
		l1[1] = (l1[5]*l1[1] + l2[5]*l2[1])/(l1[5]+l2[5])
		l1[2] = l1[2] if l1[2] < l2[2] else l2[2]
		l1[3] = l1[3] if l1[3] > l2[3] else l2[3]
		l1[5] += l2[5]

	def filter_line(self):
		lines_ = self.to_line_eq()
		self.lines = []
		while len(lines_) > 0:
			line = lines_.pop(0)
			ii = 0
			while ii < len(lines_):
				if self.compare_lines(line, lines_[ii]):
					self.merge_lines(line, lines_[ii])
					lines_.pop(ii)
				else:
					ii += 1
			if line[5] > 3:
				self.lines.append(line)

	def get_lines(self):
		self.filter_line()
		return self.lines

	class linklist(object):
		def __init__(self, region):
			self._region = region
			self._next = None
			self._previous = None

		def is_head(self):
			return 1 if self._previous is None else 0

		def is_tail(self):
			return 1 if self._next is None else 0

		def is_alone(self):
			return 1 if self._next is None and self._previous is None else 0

		@property
		def forward(self):
			return self._next

		@forward.setter
		def forward(self, x):
			if self._next is None:
				self._next = x

		@property
		def backward(self):
			return self._previous

		@backward.setter
		def backward(self, x):
			if self._previous is None:
				self._previous = x

		@property
		def region(self):
			return self._region
