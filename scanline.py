import find_pattern
import numpy as np
import cv2

class ScanLine(object):
	color_list = np.array([ [[0,0,0] ,[255,255,255]],
							], dtype = np.uint8)
	_scan_output = np.zeros((1,3))
	_region_output = np.zeros((1,4))
	_united_region = np.zeros((1,2), dtype = np.int) - 1
	scan_axis = 1
	grid_dis = 25
	step = 1
	co = 5
	def __init__(self, **kwarg):
		if "color_list" in kwarg:
			self.set_color_list(kwarg["color_list"])
		self.scan_axis = kwarg.get('scan_axis', self.scan_axis)
		self.grid_dis = kwarg.get('grid_dis', self.grid_dis)
		self.step = kwarg.get('step', self.step)
		self.co = kwarg.get('co', self.co)

	def set_color_list(self, colorList):
		if colorList.ndim != 3:
			raise ValueError("color_list's ndim != 3")
		elif colorList.shape[1] != 2 or colorList.shape[2] != 3:
			raise ValueError("color_list's shape not equal (n ,2 ,3)")
		self.color_list = colorList

	def scan_image(self, img_hsv):
		# if self.scan_output.shape != ((img.shape[self.scan_axis]//grid_dis)*(img.shape[(self.scan_axis+1)%2]//step)+1,3):
		# 	self.scan_output = np.zeros(((img.shape[self.scan_axis]//grid_dis)*(img.shape[(self.scan_axis+1)%2]//step)+1,3) dtype = np.int)
		self._scan_output = find_pattern.find_color_pattern_x(img_hsv.copy(), self.color_list, grid_dis = self.grid_dis, step = self.step, co = self.co)

	def find_region(self, img_hsv):
		self.scan_image(img_hsv)
		self._region_output = find_pattern.to_region(self._scan_output, self.scan_axis)

	def visualize_region(self, img):
		for i in self._region_output:
			# if i[3] < len(self.color_list):
			# 	img[i[1]:i[2],i[0]-(self.grid_dis/2):i[0]+(self.grid_dis)/2] = (self.color_list[i[3], 0] + self.color_list[i[3], 1])/2
			if i[3] == 0:
				img[i[1]:i[2],i[0]-(self.grid_dis/2):i[0]+(self.grid_dis)/2] = [255,255,255]
			elif i[3] == 1:
				img[i[1]:i[2],i[0]-(self.grid_dis/2):i[0]+(self.grid_dis)/2] = [0,255,0]
			else:
				img[i[1]:i[2],i[0]-(self.grid_dis/2):i[0]+(self.grid_dis)/2] = np.array([100,100,100])

	def clip_region(self, color_index):
		ii = 0
		while 1 == 1:
			temp = np.where(self._region_output[:,0] == ii)
			if len(temp[0]) == 0:
				break
			for jj in range(len(temp[0])):
				# print self._region_output[temp[0][jj],3], temp[0][jj]
				if self._region_output[temp[0][jj],3] == color_index:
					temp = temp[0][:jj]
					break
				else:
					continue
			self._region_output = np.delete(self._region_output, temp, 0)
			ii += self.grid_dis

	def connect_region(self, first, r):
		if first is None:
			return None
		while first < len(self._region_output) and self._region_output[first][2] < self._region_output[r][1] and self._region_output[first][0] + self.grid_dis == self._region_output[r][0]:
			# print "first += 1"
			first += 1
		if self._region_output[first][0] + self.grid_dis != self._region_output[r][0]:
			return None
		if self._region_output[first][1] <= self._region_output[r][2] and self._region_output[first][3] == self._region_output[r][3]:
			# print "unite\n"
			if self._united_region[first][1] == -1:
				self._united_region[first][1] = r
				self._united_region[r][0] = first
		first_ = first + 1
		while first < len(self._region_output) and self._region_output[first_][1] <= self._region_output[r][2]:
			
			if self._region_output[first_][0] + self.grid_dis != self._region_output[r][0]:
				return first
			if self._region_output[first_][2] >= self._region_output[r][1] and self._region_output[first_][3] == self._region_output[r][3]:
				# print "unite\n"
				if self._united_region[first_][1] == -1:
					self._united_region[first_][1] = r
					self._united_region[r][0] = first_
			else:
				return first
			first_ += 1
		return first

	def unite_region(self):
		self._united_region = np.zeros((self._region_output.shape[0],2), dtype = np.int) - 1
		# print self._region_output
		first = None
		last = None
		r_ = None
		for r in range(len(self._region_output)):
			if self._region_output[r][3] == 1:
				continue
			if r_ is None or self._region_output[r_][0] != self._region_output[r][0]:
				last = first
				first = r
			last = self.connect_region(last, r)
			r_ = r
		# a = np.where( self._united_region[:,1] != -1 )
		# print self._united_region

	def visualize_united_region(self, img):
		for i in range(len(self._united_region)):
			if self._united_region[i][1] != -1:
				start = self._region_output[i]
				stop = self._region_output[self._united_region[i][1]]
				cv2.line(img,(start[0],start[1]),(stop[0],stop[1]),(0,0,255),2)
				cv2.line(img,(start[0],start[2]),(stop[0],stop[2]),(0,0,255),2)
				if self._united_region[i][0] == -1:
					cv2.line(img,(start[0],start[1]),(start[0],start[2]),(0,0,255),2)
				if self._united_region[self._united_region[i][1]][1] == -1:
					cv2.line(img,(stop[0],stop[1]),(stop[0],stop[2]),(0,0,255),2)

	def link_list_to_list(self):
		point_array = []
		for ii in range(len(self._united_region)):
			temp = np.zeros((0,2), dtype = np.int)
			if self._united_region[ii][0] == -1:
				temp = np.append(temp, [[self._region_output[ii][0], (self._region_output[ii][1] + self._region_output[ii][2])/2]], axis = 0)
				ii_ = self._united_region[ii][1]
				while ii_ != -1:
					temp = np.append(temp, [[self._region_output[ii_][0], (self._region_output[ii_][1] + self._region_output[ii_][2])/2 ]], axis = 0)
					ii_ = self._united_region[ii_][1]
			if len(temp) > 1:
				point_array.append((temp, self._region_output[ii][3]))
		return point_array

	def to_line_eq(self):
		point_array = self.link_list_to_list()
		lines_ = []
		for p,color in point_array:
			x = p[:,0]
			y = p[:,1]
			A = np.vstack([x, np.ones(len(x))]).T
			m, c = np.linalg.lstsq(A, y)[0]
			lines_.append((m,c,x[0],x[-1],color))
		return lines_

	@property
	def info(self):
		a = 'scan_axis = {}\ngrid_dis = {}\nstep = {}\nco = {}\ncolor_list = {}'.format(self.scan_axis, self.grid_dis, self.step, self.co, self.color_list)
		return a





