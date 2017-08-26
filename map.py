import numpy as np 
import cv2

class Map(object):
	_component = []

	class Line(object):

		def __init__(self, start, stop, name): #format start = [x,y] ,stop = [x,t]
			self._x1 = start[0]
			self._x2 = stop[0]
			self._y1 = start[1]
			self._y2 = stop[1]
			self._a = self._x2 - self._x1
			self._b = self._y2 - self._y1
			self._c = -(self._a*self._x2) - (self._b*self._y2)
			self._name = name

		def visualize(self, img):
			cv2.line(img, (int(self._x1), int(self._y1)), (int(self._x2), int(self._y2)), (255,255,255), 5 )
			cv2.putText(img, self._name, (int((self._x1+self._x2)/2)-10, int((self._y1+self._y2)/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

		@property
		def name(self):
			return self._name

		@property
		def type(self):
			return "Line"

	def __init__(self, size): #format (w,h)
		self._w = size[0]
		self._h = size[1]

	def create_line(self, start, stop, name = None): #format start = [x,y] ,stop = [x,y]
		name_ = name if name is not None else "component" + str(len(self._component))
		line_ = self.Line(start, stop, name)
		self._component.append(line_)

	def visualize(self):
		map_ = np.zeros([int(self._h), int(self._w), 3])
		map_[:,:,1] = 255
		for c in self._component:
			c.visualize(map_)
		return map_

	@property
	def w(self):
		return self._w

	@property
	def h(self):
		return self._h

if __name__ == '__main__':
	map_ = Map([740,1040])
	
	map_.create_line([70,70], [70,970], name = "vertical_1")
	map_.create_line([670,70], [670,970], name = "vertical_2")
	map_.create_line([120,70], [120,170], name = "vertical_3")
	map_.create_line([620,70], [620,170], name = "vertical_4")
	map_.create_line([120,870], [120,970], name = "vertical_5")
	map_.create_line([620,870], [620,970], name = "vertical_6")

	map_.create_line([70,70], [670,70], name = "horizontal_1")
	map_.create_line([120,170], [620,170], name = "horizontal_2")
	map_.create_line([70,520], [670,520], name = "horizontal_3")
	map_.create_line([120,870], [620,870], name = "horizontal_4")
	map_.create_line([70,970], [670,970], name = "horizontal_5")

	img = map_.visualize()

	cv2.imshow('map', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()