from scanline import ScanLine
import numpy as np
import cv2
from util import *

class Region_reciver(object):

	def __init__(self):
		self.angle_threshold = 0.41
		self.size_ratio = 1.0

	def unite_region(self, region, connected_color):
		united_region = [ linklist(region[i]) for i in range(region.lenght)]
		first = None
		last = None
		r_ = None
		for r in range(region.lenght):
			if region[r].color != connected_color:
				continue
			if r_ is None or region[r_].column != region[r].column:
				last = first
				first = r
			last = self.connect_region(last, r, united_region, region)
			r_ = r
		return united_region

	def connect_region(self, last, r, united_region, region):
		if last is None:
			return None

		while last < region.lenght and region[last].stop < region[r].start and region[last].next_column == region[r].column:
			last += 1

		if region[last].next_column != region[r].column:
			return None

		if region[last].start <= region[r].stop and region[last].color == region[r].color:
			if self.check(united_region[last], united_region[r])==1 and united_region[last].is_tail():
				self.link_linklist(united_region[r], united_region[last])

		last_ = last + 1
		while last_ < region.lenght and region[last_].start <= region[r].stop:
			if region[last_].next_column != region[r].column:
				return last

			if region[last_].stop >= region[r].start and region[last_].color == region[r].color:
				if self.check(united_region[last_], united_region[r])==1 and united_region[last_].is_tail():
					self.link_linklist(united_region[r], united_region[last_])

			else:
				return last
			last_ += 1
		return last

	def unite_region_and_find_node(self, region, connected_color):
		united_region = [ linklist(region[i]) for i in range(region.lenght)]
		nodes = [ Node(i, region[i]) for i in range(region.lenght)]
		first = None
		last = None
		r_ = None
		for r in range(region.lenght):
			if region[r].color != connected_color:
				continue
			if r_ is None or region[r_].column != region[r].column:
				last = first
				first = r
			last = self.connect_region_and_node(last, r, united_region, nodes, region)
			r_ = r
		self.filter_node(nodes)
		return united_region, nodes

	def connect_region_and_node(self, last, r, united_region, nodes, region):
		if last is None:
			return None

		while last < region.lenght and region[last].stop < region[r].start and region[last].next_column == region[r].column:
			last += 1

		if region[last].next_column != region[r].column:
			return None

		if region[last].start <= region[r].stop and region[last].color == region[r].color:
			nodes[r].add_connected_node(nodes[last])
			nodes[last].add_connected_node(nodes[r])
			if self.check(united_region[last], united_region[r])==1 and united_region[last].is_tail():
				self.link_linklist(united_region[r], united_region[last])
		last_ = last + 1
		while last_ < region.lenght and region[last_].start <= region[r].stop:
			if region[last_].next_column != region[r].column:
				return last

			if region[last_].stop >= region[r].start:
				if region[last_].color == region[r].color:
					nodes[r].add_connected_node(nodes[last_])
					nodes[last_].add_connected_node(nodes[r])
					if self.check(united_region[last_], united_region[r])==1 and united_region[last_].is_tail():
						self.link_linklist(united_region[r], united_region[last_])
			else:
				return last
			last_ += 1
		return last

	def filter_node(self, nodes):
		i = 0
		while i < len(nodes):
			if nodes[i].is_alone():
				nodes.pop(i)
				continue
			i += 1

	def check(self, last, r):
		if last.is_head() == 0:
			v1 = last.region - last.backward.region
			v2 = r.region - last.region
			if float(r.region.stop - r.region.start) == 0 or float(last.region.stop - last.region.start) == 0:
				return 0
			ratio1 = float(last.region.stop - last.region.start)/float(r.region.stop - r.region.start)
			ratio2 = float(r.region.stop - r.region.start)/float(last.region.stop - last.region.start)
			ratio = max(ratio1, ratio2)
			if angle_between(v1[:2], v2[:2]) < self.angle_threshold:
				if self.size_ratio is None:
					return 1
				elif ratio < 1.0 + self.size_ratio:
					return 1
				return 0
			else:
				return 0
		else:
			if float(r.region.stop - r.region.start) == 0 or float(last.region.stop - last.region.start) == 0:
				return 0
			ratio1 = float(last.region.stop - last.region.start)/float(r.region.stop - r.region.start)
			ratio2 = float(r.region.stop - r.region.start)/float(last.region.stop - last.region.start)
			ratio = max(ratio1, ratio2)
			if self.size_ratio is None:
				return 1
			elif ratio < 1.0 + self.size_ratio:
				return 1
			return 0

	def link_list_to_list(self, united_region):
		point_array = []
		for ii in united_region:
			temp = np.zeros((0,2), dtype = np.int)
			if ii.is_head() and ii.is_alone() == 0:
				temp = np.append(temp, [[ii.region.column, ii.region.middle]], axis = 0)
				ii_ = ii.forward
				while ii_.is_tail() == 0:
					temp = np.append(temp, [[ii_.region.column, ii_.region.middle]], axis = 0)
					ii_ = ii_.forward
				temp = np.append(temp, [[ii_.region.column, ii_.region.middle]], axis = 0)
				if temp.shape[0] > 1:
					point_array.append((temp, ii.region.color))
		return point_array

	def link_linklist(self, forward, backward):
		forward.backward = backward
		backward.forward = forward

	def visualize_united_region(self, img, united_region, axis):
		for i in united_region:
			if i.is_tail() == 0:
				start = i.region
				stop = i.forward.region
				p1 = (start.column,start.start) if axis == 1 else (start.start,start.column)
				p2 = (stop.column,stop.start) if axis == 1 else (stop.start,stop.column)
				p3 = (start.column,start.stop) if axis == 1 else (start.stop,start.column)
				p4 = (stop.column,stop.stop) if axis == 1 else (stop.stop,stop.column)
				cv2.line(img, p1, p2, (0,0,255), 2)
				cv2.line(img, p3, p4, (0,0,255), 2)
				if i.is_head():
					cv2.line(img, p1, p3, (0,0,255), 2)
				if i.forward.is_tail():
					cv2.line(img, p2, p4, (0,0,255), 2)

	def visualize_node(self, img, nodes, axis, circle_size = 3, color = [255, 0, 255]):
		for n in nodes:
			p = (n.data.column,n.data.middle) if axis == 1 else (n.data.middle, n.data.column)
			cv2.circle(img, p, circle_size * n.number_of_connected_node, color, -1)
			for cn in n.connected_node:
				p1 = (cn.data.column,cn.data.middle) if axis == 1 else (cn.data.middle,cn.data.column)
				p2 = (n.data.column,n.data.middle) if axis == 1 else (n.data.middle,n.data.column)
				cv2.line(img, p1, p2, color, 2)

	def set_size_ratio(self, size_ratio):
		if size_ratio is None:
			self.size_ratio = None
		elif size_ratio < 0:
			self.size_ratio = -size_ratio
		else:
			self.size_ratio = size_ratio

	def set_angle_threshold(self, angle_threshold):
		self.angle_threshold = angle_threshold

class linklist(object):
	def __init__(self, region):
		self._region = region
		self._next = None
		self._previous = None

	def is_head(self):
		return 1 if self._previous is None else 0

	def is_tail(self):
		if self._next is not None and self._next._previous is None:
			print "maybe this a bug."
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

class Node(object):
	def __init__(self, id_, data):
		self.id = id_
		self.data = data
		self.connected_node = []
		self.connected_node_id = []

	def is_alone(self):
		return 1 if len(self.connected_node_id) == 0 else 0

	def add_connected_node(self, node):
		if node.id not in self.connected_node_id:
			self.connected_node.append(node)
			self.connected_node_id.append(node.id)

	@property
	def number_of_connected_node(self):
		return len(self.connected_node_id)