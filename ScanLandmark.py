import numpy as np
import cv2
from scanline import ScanLine
from line_provider import Line_provider
from Goal_provider import Goal_provider
from cross_provider import Cross_provider
import time

class ScanLandmark(object):
	class Color(object):
		def __init__(self, id_, name, RenderColor_RGB, color_min, color_max):
			self.id = id_
			self.index = id_ - 1
			self.name = name
			self.RenderColor_RGB = RenderColor_RGB
			self.color_min = color_min
			self.color_max = color_max

	def __init__(self, config, **kwarg):
		'''
		config is a configobj object from color_config.
		if is_do_horizontal is False, get_goals always return None.
		'''
		line_angleThreshold = kwarg.get("line_angleThreshold", 0.4)
		line_sizeRatio = kwarg.get("line_sizeRatio", 0.5)
		goal_angleThreshold = kwarg.get("goal_angleThreshold", 0.4)
		goal_sizeRatio = kwarg.get("goal_sizeRatio", 0.3)
		verticalGridDis = kwarg.get("verticalGridDis", 20)
		horizontalGridis = kwarg.get("horizontalGridis", 20)
		verticalCO = kwarg.get("verticalCO",2)
		horizontalstep = kwarg.get("horizontalstep",5)
		verticalstep = kwarg.get("verticalstep", 1)
		self.frechet_d_thr = kwarg.get("frechet_d_thr", d=20)
		self.is_do_horizontal = kwarg.get("is_do_horizontal", True)
		self.img_w = kwarg.get("img_w", d=640)
		self.img_h = kwarg.get("img_h", d=480)
		self.cross_check_tolerance = kwarg.get("cross_check_tolerance", d=40)
		self.config = config
		self.color_list = None
		self.color_dict = {}
		self.__create_color_list()

		self.vertical_scan = ScanLine(color_list = self.color_list, 
							grid_dis = verticalGridDis,
							scan_axis = 1,
							co = verticalCO,
							step = verticalstep)
		self.horizontal_scan = ScanLine(color_list = self.color_list,
										grid_dis = horizontalGridis,
										scan_axis = 0,
										co = 1,
										step = horizontalstep1)
		self.line_p = Line_provider(angle_threshold=line_angle_threshold, size_ratio=line_sizeRatio)
		self.cross_p = Cross_provider()
		self.goal_p = Goal_provider(goal_angleThreshold, goal_sizeRatio, h2w_ratio= 2.0, distance_thr= 75)

	def __create_color_list(self):
		colorDefinition = self.config["ColorDefinitions"]
		self.color_list =np.zeros((len(colorDefinition),2,3),dtype = np.uint8)
		for i,(_,color) in enumerate(colorDefinition.items()):
			id_ = color["id"]
			name = color["name"]
			RenderColor_RGB = eval(color["RenderColor_RGB"])
			H_Max = int(color["H_Max"])
			H_Min = int(color["H_Min"])
			S_Max = int(color["S_Max"])
			S_Min = int(color["S_Min"])
			V_Max = int(color["V_Max"])
			V_Min = int(color["V_Min"])
			self.color_list[i,0] = np.array([H_Min,S_Min,V_Min], dtype = np.uint8)
			self.color_list[i,1] = np.array([H_Max,S_Max,V_Max], dtype = np.uint8)
			self.color_dict[name] = self.Color(id_, name, RenderColor_RGB, np.array([H_Min,S_Min,V_Min], dtype=np.uint8), np.array([H_Max,S_Max,V_Max], dtype=np.uint8))

	def do_scan_image(self, img, cvt2hsv=True, horizon = 0, end_scan = -1):
		'''Must be called first. This function scan image with ScanLine. Do vertical scan first
		then horizontal scan if is_do_horizontal is True. This function return a boundary of green 
		area in an image.
		img is image to be scaned.
		cvt2hsv if True, will convert img from rgb to hsv first.
		horizon can be calculated by tilt angle and/or robot kinematic and/or infomation from IMU
		vertical ScanLine will start scan at y = horizon and end at y = end_scan if end_scan = -1 it will end
		at bottom of image.
		'''
		img_hsv = img.copy() if not cvt2hsv else cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		self.vertical_scan.find_region(img_hsv, horizon= horizon, end_scan= end_scan)
		boundary = self.vertical_scan.clip_region(self.color_dict["green"].index)
		horizon_end_scan = max(boundary,   key = lambda x : x[1])[1]

		boundary.append(np.array([img_hsv.shape[1], img_hsv.shape[0]]))
		boundary.append(np.array([0,img_hsv.shape[0]]))
		if self.is_do_horizontal:
			cv2.drawContours(img_hsv, [cv2.convexHull(np.array([boundary]))], 0, (0,0,0), -1)
			self.horizontal_scan.find_region(img_hsv, horizon= 0, end_scan= horizon_end_scan)

		return np.array(boundary)

	def get_vertical_regions(self):
		'''
		Must be called after do_scan_image. Return list of instance of region class
		for vertical scanline(See definition at scanline.py).
		'''
		return self.vertical_scan.get_regions()

	def get_vertical_regions(self):
		'''
		Must be called after do_scan_image. Return list of instance of region class
		for vertical scanline(See definition at scanline.py). If pass is_do_horizon = False 
		to ScanLandmark constructor, this function always return None. 
		'''
		return self.horizontal_scan.get_regions() if self.is_do_horizontal else None

	def get_lines(self):
		'''Must be called after do_scan_image. This function find a line field in an image.
		This function return list of instance of Line class(See definition at line_provider.py).
		'''
		self.line_p.receive_region(self.vertical_scan, connected_color=self.color_dict["white"].index)
		return self.line_p.get_lines(axis= 1, frechet_distance= self.frechet_d_thr)

	def get_crosses(self):
		''' Must be called after get_lines. This function find a cross in an image.
		This function return list of instance of Cross class(See definition at cross_provider.py).
		'''
		self.cross_p.receive_line(self.line_p, max_h=self.img_h, max_w=self.img_w, tolerance= self.cross_check_tolerance)
		return self.cross_p.get_point()

	def get_goals(self, boundary):
		''' Must be called after do_scan_image. If pass is_do_horizon = False to ScanLandmark constructor, 
		this function always return None.Otherwise, return two point for goal position. 
		'''
		if self.is_do_horizontal:
			self.goal_p.receive_region(self.horizontal_scan, connected_color=self.color_dict["white"].index)
			return self.goal_p.get_filtred_Squar(boundary= cv2.convexHull(np.array([boundary])))
		return None

	def visualize_vertical_region(self, img):
		self.vertical_scan.visualize_region(img)

	def visualize_horizontal_region(self, img):
		self.horizontal_scan.visualize_region(img)

	def visualize_node(self, img):
		self.line_p.visualize_node(img)