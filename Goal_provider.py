import numpy as np
import cv2
from util import *
from region_reciever import Region_reciver
from scanline import ScanLine

class Goal_provider( object ):
	def __init__(self, angle_threshold, size_ratio, h2w_ratio = 2.0, distance_thr = 75):
		self.angle_threshold = angle_threshold
		self.size_ratio = size_ratio
		self._region = None
		self._united_region = None
		self._h2w_ratio = h2w_ratio
		self._distance_thr = distance_thr
		self.goal = []
		self.gKernel = cv2.getGaussianKernel(7, 0).reshape(-1)
		self.kernel = np.array([10, 0, -10], dtype = np.float64)

	def receive_region(self, img,region, color_dict, checkStep = 10, checkIterate = 20):
		self._region = region
		self.filter( img, color_dict, checkStep, checkIterate)
		# self._united_region = super( Goal_provider, self).unite_region(self._region, connected_color)

	# def visualize_united_region(self, img, axis = 1):
	# 	super( Goal_provider, self).visualize_united_region(img, self._united_region, axis)

	# def LinkList2Squar(self):
	# 	point_array = []
	# 	for i in self._united_region:
	# 		temp = np.zeros((2,2), dtype = np.int)
	# 		if i.is_head() and i.is_alone() == 0:
	# 			temp[0] = [i.region.column, i.region.start]
	# 			i_ = i.forward
	# 			while i_.is_tail() == 0:
	# 				i_ = i_.forward
	# 			temp[1] = [i_.region.column, i_.region.stop]
	# 			point_array.append(temp)
	# 	return point_array

	def filter(self, img,color_dict, checkStep, checkIterate):
		self.goal[:] = []
		for r in range(0,self._region.lenght):
			if self._region[r].color == color_dict["white"].index:
				middle_point = self._region[r].middle
				pl_x = self._region[r].start[0] 
				pr_x = self._region[r].stop[0]
				if pr_x - pl_x > 120:
					mids = [(pl_x+8, middle_point[1]), (pr_x-8, middle_point[1])]
					for mid in mids:
						self.__check(img, color_dict, mid, checkStep, checkIterate, 16)
				else:
					self.__check(img, color_dict, middle_point, checkStep, checkIterate, pr_x - pl_x)

	def __check(self, img, color_dict,middle_point,checkStep, checkIterate, width):
		check_X = middle_point[0]
		pl_x, pr_x = check_X - width/2, check_X + width/2
		temp = []
		for i in range(1, checkIterate + 1):
			check_Y = middle_point[1] - checkStep*i
			if check_Y < 0:
				break
			if classify_color(img[check_Y,check_X], color_dict["white"].color_max, color_dict["white"].color_min):
				l_x, r_x = self.__search_goal_edge(img, check_X, check_Y)
				if l_x <= pr_x and r_x >= pl_x:
					if l_x == r_x or pr_x == pl_x:
						break
					ratio = max((r_x - l_x)/(pr_x - pl_x),(pr_x - pl_x)/(r_x - l_x))
					if ratio > 1+self.size_ratio:
						pass
					else:
						temp.append(((r_x + l_x)/2, check_Y))
						pr_x = r_x
						pl_x = l_x
						check_X = (pr_x + pl_x) / 2
				else:
					break
			if len(temp) > 1:
				goal_base = self.__search_actual_goalBase(img, middle_point[0], middle_point[1], color_dict)
				if goal_base[1] - temp[-1][1] > 20:
					temp.append(goal_base)
					self.goal.append([])
					self.goal[-1] = temp

	def __search_goal_edge(self,img,x,y):
		max_h = img.shape[0]
		max_w = img.shape[1]
		col = img[y,:,2]
		a = np.convolve(col, self.gKernel, mode = 'same')
		a = np.absolute(np.convolve(a, self.kernel, mode = 'full'))[1:-1]
		a = np.convolve(a, self.gKernel, mode = 'same')
		a = np.clip(a, 0, 100)
		a = (a/100)*255
		left_x = x
		right_x = x
		pre_val = 0
		state_climb = 0
		state_max = 0
		for left_x in range(x-1,0,-1):
			if a[left_x] > pre_val:
				state_max = 1
				pre_val = a[left_x]
			elif state_max:
				if a[left_x+1] > 100:
					left_x = left_x+1
					break
				else:
					state_max = 0
		pre_val = 0
		state_max = 0
		for right_x in range(x+1,max_w,1):
			if a[right_x] > pre_val:
				state_max = 1
				pre_val = a[right_x]
			elif state_max:
				if a[right_x-1] > 100:
					right_x = right_x - 1
					break
				else:
					state_max = 0
		return (left_x, right_x)

	def __search_actual_goalBase(self, img, X, Y, color_dict):
		max_h = img.shape[0]
		for y in range(Y,max_h):
			# if (color_dict["white"].color_min > img[y,X]).any() or (img[y,X] > color_dict["white"].color_max).any():
			if not classify_color(img[y,X], color_dict["white"].color_max, color_dict["white"].color_min):
				return (X,y)
		return (X,y)

	def get_goal(self):
		goal = self.__goal_list_to_class()
		return  goal

	def __goal_list_to_class(self):
		goal = []
		for g in self.goal:
			base = g[-1]
			top = g[-2]
			width = sum(i for i,_ in g) / len(g)
			goal.append(Goal(base, top, width))
		return goal

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

class Goal(object):
	LEFT = 0
	RIGHT = 1
	UNKNOWN = 2

	def __init__(self, goal_base, top, type_ = UNKNOWN):
		self.goal = (goal_base, top, type_)

	@property
	def goal_base(self):
		return self.goal[0]

	@property
	def top(self):
		return self.goal[1]

	@property
	def high(self):
		return self.goal[1][1] - self.goal[0][1]

	@property
	def type(self):
		return self.goal[2]