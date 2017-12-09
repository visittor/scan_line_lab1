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

	def find_region(self, img_hsv, horizon = 0, end_scan = -1):
		self.scan_image(img_hsv, horizon, end_scan)
		self._region_output = find_pattern.to_region(self._scan_output, self.scan_axis)

	def visualize_region(self, img):
		a = self._visualize_region_axis_0(img) if self.scan_axis == 0 else self._visualize_region_axis_1(img)
		
	def _visualize_region_axis_0(self, img):
		for i in self._region_output:
			if i == [0, 0, 0]:
				break
			if i[3] == 0:
				img[i[0]-(self.grid_dis/5):i[0]+(self.grid_dis/5),i[1]:i[2]] = [255,255,255]
			elif i[3] == 1:
				img[i[0]-(self.grid_dis/5):i[0]+(self.grid_dis/5),i[1]:i[2]] = [0,255,0]
			elif i[3] == 2:
				img[i[0]-(self.grid_dis/5):i[0]+(self.grid_dis/5),i[1]:i[2]] = np.array([0,0,0])
			else:
				img[i[0]-(self.grid_dis/5):i[0]+(self.grid_dis/5),i[1]:i[2]] = np.array([100,100,100])

	def _visualize_region_axis_1(self, img):
		for i in self._region_output:
			if i[3] == 0:
				img[i[1]:i[2],i[0]-(self.grid_dis/5):i[0]+(self.grid_dis)/5] = [255,255,255]
			elif i[3] == 1:
				img[i[1]:i[2],i[0]-(self.grid_dis/5):i[0]+(self.grid_dis)/5] = [0,255,0]
			elif i[3] == 2:
				img[i[1]:i[2],i[0]-(self.grid_dis/5):i[0]+(self.grid_dis)/5] = np.array([0,0,0])
			else:
				img[i[1]:i[2],i[0]-(self.grid_dis/5):i[0]+(self.grid_dis)/5] = np.array([100,100,100])

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
		return boundary

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