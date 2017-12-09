from scanline import ScanLine
from region_reciever import Region_reciver
import numpy as np
import cv2
from util import *

class Line_provider(Region_reciver):
	_united_region = []
	_region = None
	lines = []

	def __init__(self, angle_threshold = 0.51, size_ratio=1.0):
		super( Line_provider, self).__init__()
		super( Line_provider, self).set_size_ratio(size_ratio)
		super( Line_provider, self).set_angle_threshold(angle_threshold)
		self._united_region = []
		self.nodes = []
		self._region = None
		self.__grid_dis = None
		self.lines = []
		self.crossing = []

	def receive_region(self, region, connected_color = 0):
		if region.__class__ == ScanLine:
			self._region = region
			self._united_region, self.nodes = super(Line_provider, self).unite_region_and_find_node(self._region, connected_color)
		else:
			raise ValueError("region is not class ScanLine.region")

	def append(self, region, connected_color = 0):
		united_region_ = self._united_region 
		self._region = region
		self.unite_region(connected_color)
		self._united_region.extend(united_region_)

	def visualize_united_region(self, img, axis = 1):
		super( Line_provider, self).visualize_united_region(img, self._united_region, axis)

	def visualize_node(self, img, axis = 1):
		super( Line_provider, self).visualize_node(img, self.nodes, axis)

	def link_list_to_list(self):
		return super( Line_provider, self ).link_list_to_list(self._united_region)

	def node_to_list(self, nodes):
		# visited_nodes = np.zeros(len(nodes), dtype = np.uint8)
		visited_nodes = []
		line_array = []
		point_array = []
		i = 0
		while len(visited_nodes) != len(nodes):
			if nodes[i].id in visited_nodes:
				pass
				# print "pass"
			elif nodes[i].number_of_connected_node == 1:
				point_array.append(nodes[i])
				visited_nodes.append(nodes[i].id)
				# print "1 node"
				self.search_node(nodes[i], visited_nodes, line_array, point_array)
				i += 1
			elif nodes[i].number_of_connected_node == 2:
				if self.check_node_angle(nodes[i]):
					point_array.append(nodes[i])
					visited_nodes.append(nodes[i].id)
					# print "2 	node"
					self.search_node(nodes[i], visited_nodes, line_array, point_array)
			else:
				point_array.append(nodes[i])
				visited_nodes.append(nodes[i].id)
				# print "3 node"
				self.search_node(nodes[i], visited_nodes, line_array, point_array)
			i = (i+1)%len(nodes)
		# print line_array
		return point_array, line_array

	def search_node(self, node, visited_nodes, line_array, point_array):
		for next_node in node.connected_node:
			if next_node.id in visited_nodes:
				continue
			line_array.append([np.zeros((0,2), dtype = np.int), node.data.color])
			# temp = np.zeros((0,2), dtype = np.int)
			while next_node.id not in visited_nodes:
				if next_node.number_of_connected_node == 1:
					point_array.append(next_node)
					visited_nodes.append(next_node.id)
					# print node.id,"Found node 1"
					break
				elif next_node.number_of_connected_node == 2:
					if self.check_node_angle(next_node):
						# print node.id,"Found node 2"
						point_array.append(next_node)
						visited_nodes.append(next_node.id)
						self.search_node(next_node, visited_nodes, line_array, point_array)
						break
					else:
						# print "Found line"
						line_array[-1][0] = np.append(line_array[-1][0], [[next_node.data.column, next_node.data.middle]], axis = 0)
						# print node.id, "...", next_node.id, line_array[-1]
						# temp = np.append(temp, [[next_node.data.column, next_node.data.middle]], axis = 0)
						# print node.id, "...", id(temp)
						visited_nodes.append(next_node.id)
						next_node = next_node.connected_node[0] if next_node.connected_node[0].id not in visited_nodes else next_node.connected_node[1]
				else:
					point_array.append(next_node)
					visited_nodes.append(next_node.id)
					self.search_node(next_node, visited_nodes, line_array, point_array)
					# print node.id,"Found node 3"
					break
			# if temp.shape[0] > 1:
			# 	print "temp"
			# 	line_array.append((temp, next_node.data.color))
		# print "finish one function"

	def check_node_angle(self, node):
		v1 = node.connected_node[0].data - node.data
		v2 = node.connected_node[1].data - node.data
		ang = angle_between(v1[:2], v2[:2])
		# print ang
		if ang < 3.14 - self.angle_threshold :
			return 1
		return 0

	def to_line_eq(self, axis = 1):
		# point_array = self.link_list_to_list()
		crossing,point_array = self.node_to_list(self.nodes)
		lines_ = []
		for p,color in point_array:
			x = p[:,0] if axis == 1 else p[:,1]
			y = p[:,1] if axis == 1 else p[:,0]
			A = np.vstack([x, np.ones(len(x))]).T
			try:
				m, c = np.linalg.lstsq(A, y)[0]
				lines_.append([m,c,x[0],x[-1],color,p.shape[0]])
			except ValueError:
				pass
		return lines_, crossing

	def compare_lines(self, l1, l2, frechet_d_thr = 15):
		m1 = [np.array([ 0, l1[1] ]), np.array([100, 100*l1[0]+l1[1] ]) ]
		m2 = [np.array([ 0, l2[1] ]), np.array([100, 100*l2[0]+l2[1] ]) ]
		frechet_d = frechet_distance(m1, m2)
		if frechet_d < frechet_d_thr:
			return 1
		return 0 

	def merge_lines(self, l1, l2):
		l1[0] = (l1[5]*l1[0] + l2[5]*l2[0])/(l1[5]+l2[5])
		l1[1] = (l1[5]*l1[1] + l2[5]*l2[1])/(l1[5]+l2[5])
		l1[2] = l1[2] if l1[2] < l2[2] else l2[2]
		l1[3] = l1[3] if l1[3] > l2[3] else l2[3]
		l1[5] += l2[5]

	def filter_line(self, axis = 1, frechet_d_thr = 15):
		lines_,self.crossing = self.to_line_eq( axis = axis)
		self.lines = []
		while len(lines_) > 0:
			line = lines_.pop(0)
			ii = 0
			while ii < len(lines_):
				if self.compare_lines(line, lines_[ii], frechet_d_thr = frechet_d_thr):
					self.merge_lines(line, lines_[ii])
					lines_.pop(ii)
				else:
					ii += 1
			if line[5] > 3:
				self.lines.append(line)

	def make_line(self, axis = 1, frechet_d_thr = 15):
		self.filter_line(axis = axis, frechet_d_thr = frechet_d_thr)

	def get_lines_array(self, axis = 1, frechet_d_thr = 15):
		self.filter_line(axis = axis, frechet_d_thr = frechet_d_thr)
		return self.lines

	def get_lines(self, axis = 1, frechet_distance = 15):
		self.filter_line(axis=axis, frechet_d_thr= frechet_distance)
		return [Line(l) for l in self.lines]

	@property
	def lenght(self):
		return len(self.lines)
		
	def __getitem__(self, index):
		return Line(self.lines[index])

class Line(object):

	def __init__(self, line_array):
		self._line_array = line_array

	@property
	def m(self):
		return self._line_array[0]

	@property
	def c(self):
		return self._line_array[1]

	@property
	def start(self):
		y = self._line_array[0]*self._line_array[2] + self._line_array[1]
		return np.array([self._line_array[2], y])

	@property
	def stop(self):
		y = self._line_array[0]*self._line_array[3] + self._line_array[1]
		return np.array([self._line_array[3], y])

	@property
	def startX(self):
		return self._line_array[2]

	@property
	def stopX(self):
		return self._line_array[3]

	@property
	def color(self):
		return self._line_array[4]

	@property
	def vote(self):
		return self._line_array[5]