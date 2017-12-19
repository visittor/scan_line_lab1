import find_pattern
import numpy as np
import cv2
from util import *

class ScanLine(object):
	
	def __init__(self, **kwarg):
		# self._region_output = np.zeros((1,4))
		# self.color_list = np.array([ [[0,0,0] ,[255,255,255]],
		# 					], dtype = np.uint8)
		# self._scan_output = np.zeros((1,3))
		# self.scan_axis = 1
		# self.grid_dis = 25
		# self.step = 1
		# self.co = 10
		if "color_list" in kwarg:
			self.set_color_list(kwarg["color_list"])
		else:
			self.set_color_list(np.array([ [[0,0,0] ,[255,255,255]],], dtype = np.uint8))

		self.scan_axis = kwarg.get('scan_axis', 1)
		self.grid_dis = kwarg.get('grid_dis', 25)
		self.step = kwarg.get('step', 1)
		self.co = kwarg.get('co', 10)

	def set_color_list(self, colorList):
		if colorList.ndim != 3:
			raise ValueError("color_list's ndim != 3")
		elif colorList.shape[1] != 2 or colorList.shape[2] != 3:
			raise ValueError("color_list's shape not equal (n ,2 ,3)")
		self.color_list = colorList

	def scan_image(self, img_hsv, horizon, end_scan):
		self._scan_output = find_pattern.find_color_pattern_x(img_hsv.copy(), self.color_list, grid_dis = self.grid_dis, step = self.step, co = self.co, horizon = horizon, end_scan = end_scan) if self.scan_axis == 1 else find_pattern.find_color_pattern_y(img_hsv.copy(), self.color_list, grid_dis = self.grid_dis, step = self.step, co = self.co, horizon = horizon, end_scan = end_scan)
		# print self._scan_output.shape
		y = 0
		# for i in range(len(self._scan_output)):
		# 	y += 1
	def visualize_scan_line(self, img):
		for i in self._scan_output:
			if img.shape[2] == 3 :
				if i[2] == 0:
					img[i[0]-0:i[0]+1, i[1]-0:i[1]+1] = [255, 255, 255]
				elif i[2] == 1:
					img[i[0]-0:i[0]+1, i[1]-0:i[1]+1] = [0, 255, 0]
				else:
					img[i[0]-0:i[0]+1, i[1]-0:i[1]+1] = [100, 100,100]
			else:
				img[i[0], i[1]] = 255

	def find_region(self, img_hsv, horizon = 0, end_scan = -1, minPix = 15):
		self.scan_image(img_hsv, horizon, end_scan)
		self._region_output = find_pattern.to_region(self._scan_output, self.scan_axis, min_pixel = minPix)

	def visualize_region(self, img, color_obj):
		a = self._visualize_region_axis_0(img, color_obj) if self.scan_axis == 0 else self._visualize_region_axis_1(img, color_obj)
		
	def _visualize_region_axis_0(self, img, color_obj):
		for i in self._region_output:
			for color in color_obj:
				if i[3] == color.index:
					img[i[0]-(self.grid_dis/5):i[0]+(self.grid_dis/5),i[1]:i[2]] = color.RenderColor_RGB

	def _visualize_region_axis_1(self, img, color_obj):
		for i in self._region_output:
			for color in color_obj:
				if i[3] == color.index:
					img[i[1]:i[2],i[0]-(self.grid_dis/5):i[0]+(self.grid_dis)/5] = color.RenderColor_RGB

	def clip_region(self, color_index):
		ii = 0
		boundary = []
		while 1 == 1:
			temp = np.where(self._region_output[:,0] == ii)
			if len(temp[0]) == 0:
				break
			for jj in range(len(temp[0])):
				if self._region_output[temp[0][jj],3] == color_index:
					point = np.array([ self._region_output[temp[0][jj],1], self._region_output[temp[0][jj],0]]) if self.scan_axis == 0 else np.array([ self._region_output[temp[0][jj],0], self._region_output[temp[0][jj],1] ])
					boundary.append(point)
					temp = temp[0][:jj]
					break
				else:
					continue
			self._region_output = np.delete(self._region_output, temp, 0)
			ii += self.grid_dis
		return np.array(boundary)

	def get_regions(self):
		return [self.region(self._region_output[i], self.grid_dis) for i in range(len(self._region_output))]

	def get_numpy_regions(self):
		return self._region_output.copy(), self.grid_dis

	def __getitem__(self, index):
		return region(self._region_output[index], self.grid_dis)

	@property
	def info(self):
		a = 'scan_axis = {}\ngrid_dis = {}\nstep = {}\nco = {}\ncolor_list = {}'.format(self.scan_axis, self.grid_dis, self.step, self.co, self.color_list)
		return a

	# @property
	def r(self, minPix):
		self._region_output = self._region_output[np.where(self._region_output[:,2]-self._region_output[:,1] < minPix)]

	@property
	def lenght(self):
		return len(self._region_output)

class region(object):
	def __init__(self, region, grid_dis):
		self.region = region
		self.grid_dis = grid_dis
	@property
	def column(self):
		return self.region[0]

	@property
	def next_column(self):
		return self.region[0] + self.grid_dis

	@property
	def start(self):
		return self.region[1]

	@property
	def stop(self):
		return self.region[2]

	@property
	def middle(self):
		return (self.region[1] + self.region[2])/2

	@property
	def color(self):
		return self.region[3]

	def __sub__(self, b):
		a = self.region[:3] - b.region[:3]
		a[1] = (a[1] + a[2])/2
		return a[:2]

class scanline_polygon(object):
	def __init__(self, **kwarg):
		# self._region_output = np.zeros((1,4))
		# self.color_list = np.array([ [[0,0,0] ,[255,255,255]],
		# 					], dtype = np.uint8)
		# self._scan_output = np.zeros((1,3))
		# self.scan_axis = 1
		# self.grid_dis = 25
		# self.step = 1
		# self.co = 10
		if "color_list" in kwarg:
			self.set_color_list(kwarg["color_list"])
		else:
			self.set_color_list(np.array([ [[0,0,0] ,[255,255,255]],], dtype = np.uint8))

		self.step = kwarg.get('step', 1)
		self.minPix = kwarg.get('minPix', 10)

	def set_color_list(self, colorList):
		if colorList.ndim != 3:
			raise ValueError("color_list's ndim != 3")
		elif colorList.shape[1] != 2 or colorList.shape[2] != 3:
			raise ValueError("color_list's shape not equal (n ,2 ,3)")
		self.color_list = colorList

	def scan_image(self, img_hsv, boundary):
		self._scan_output = find_pattern.find_color_pattern_polygon(img_hsv, self.color_list, boundary + 5, step = self.step)

	def find_region(self, img_hsv, boundary, end_scan = -1):
		self.scan_image(img_hsv, boundary)
		self._region_output = find_pattern.to_region_from_polygon(self._scan_output, minpix = self.minPix)

	def visualize(self,img,color_dict):
		# find_pattern.visualize_polygon_scanline(img, self._scan_output, color_dict)
		for r in self._region_output:
			for color in color_dict.values():
				if r[4] == color.index:
					cv2.line(img, (r[1], r[0]), (r[3], r[2]), color.RenderColor_RGB, 2)

	def __getitem__(self, index):
		return self.region_for_polygon(self._region_output[index])

	@property
	def lenght(self):
		return len(self._region_output)

	class region_for_polygon(object):
		def __init__(self, region):
			self.region = region

		@property
		def start(self):
			return (self.region[1],self.region[0])

		@property
		def stop(self):
			return (self.region[3],self.region[2])

		@property
		def middle(self):
			return ((self.region[1] + self.region[3])/2, (self.region[0] + self.region[2])/2)
			# return (self.region[3],self.region[2])

		@property
		def color(self):
			return self.region[4]

		def __sub__(self, b):
			a = self.region[:4] - b.region[:4]
			return a[:4]