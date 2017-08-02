import find_pattern
import numpy as np
import cv2
from util import *

class ScanLine(object):
	color_list = np.array([ [[0,0,0] ,[255,255,255]],
							], dtype = np.uint8)
	_scan_output = np.zeros((1,3))
	_region_output = np.zeros((1,4))
	scan_axis = 1
	grid_dis = 25
	step = 1
	co = 10
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
		self._scan_output = find_pattern.find_color_pattern_x(img_hsv.copy(), self.color_list, grid_dis = self.grid_dis, step = self.step, co = self.co) if self.scan_axis == 1 else find_pattern.find_color_pattern_y(img_hsv.copy(), self.color_list, grid_dis = self.grid_dis, step = self.step, co = self.co)

	def visualize_scan_line(self, img):
		for i in self._scan_output:
			if img.shape[2] == 3 :
				img[i[0], i[1]] = [255, 255, 255] if i[2] == 0 else [0, 255, 0]
				img[i[0], i[1]] = [0, 255, 0] if i[2] == 1 else [100, 100,100]
			else:
				img[i[0], i[1]] = 255

	def find_region(self, img_hsv):
		self.scan_image(img_hsv)
		self._region_output = find_pattern.to_region(self._scan_output, self.scan_axis)

	def visualize_region(self, img):
		a = self._visualize_region_axis_0(img) if self.scan_axis == 0 else self._visualize_region_axis_1(img)
		
	def _visualize_region_axis_0(self, img):
		for i in self._region_output:
			if i == [0, 0, 0]:
				break
			if i[3] == 0:
				img[i[0]-5:i[0]+5,i[1]:i[2]] = [255,255,255]
			elif i[3] == 1:
				img[i[0]-5:i[0]+5,i[1]:i[2]] = [0,255,0]
			else:
				img[i[0]-5:i[0]+5,i[1]:i[2]] = np.array([100,100,100])

	def _visualize_region_axis_1(self, img):
		for i in self._region_output:
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
				if self._region_output[temp[0][jj],3] == color_index:
					temp = temp[0][:jj]
					break
				else:
					continue
			self._region_output = np.delete(self._region_output, temp, 0)
			ii += self.grid_dis

	def __getitem__(self, index):
		return self.region(self._region_output[index], self.grid_dis)

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
			return self.region - b.region

	@property
	def info(self):
		a = 'scan_axis = {}\ngrid_dis = {}\nstep = {}\nco = {}\ncolor_list = {}'.format(self.scan_axis, self.grid_dis, self.step, self.co, self.color_list)
		return a

	@property
	def lenght(self):
		return len(self._region_output)





