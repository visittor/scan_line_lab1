import numpy as np 
import cv2
import random


class Particle_Flock(object):

	def __init__(self, n_particle = 100, dimention = 3, boundary = None, cost_function = None): #for 2d field use dimention = 3 for x, y and rotation
		self._cost_function = cost_function
		self._n_particle = n_particle
		self._dimention = dimention
		if boundary is not None:  #boundary must be array like with dimention (dimention, 2)
			if len(boundary) != self._dimention:
				raise ValueError("boundary must be array like with dimention (dimention, 2)")
			self._boundary = boundary
		self._flock = np.zeros([n_particle, dimention+1], dtype = np.float) # [x1, x2, x3, ... , xn, weight_of_this_particle]

	def release_particles(self):
		if len(self._boundary) != self._dimention:
			raise ValueError("boundary must be array like with dimention (dimention, 2)")
		random.seed()
		for i in range(self._n_particle):
			for j in range(self._dimention):
				self._flock[i,j] = ((self._boundary[j][1]-self._boundary[j][0])*random.random()) + self._boundary[j][0]

	def calculate_cost(self):
		for i in self._flock:
			i[-1] = self._cost_function(i[:-1])
		self._flock[:,self._dimention] = self._flock[:,self._dimention]/sum(self._flock[:,self._dimention]) # nomalize cost

	def attached_cost_function(self, cost_function):
		self._cost_function = cost_function

	def resample(self, radius, spare_percentage = 10):
		spare_percentage = spare_percentage%100
		spare = (self._n_particle*spare_percentage)/100
		if len(radius) != self._dimention:
			raise ValueError("radius must be array like with dimention (dimention, 1)")
		distibution = [0]*self._n_particle
		np.random.seed()
		for n in range(0, self._n_particle-spare):
			random_num = np.random.random_sample()
			a = 0.0
			for i in range(self._n_particle):
				a += self._flock[i,-1]
				if a >= random_num :
					distibution[i] += 1
					break

		temp = np.zeros([self._n_particle, self._dimention+1], dtype = np.float)
		index = 0
		for n,p in zip(distibution, self._flock):
			for ii in range(n):
				delta = np.random.normal(size = self._dimention)
				delta = delta*radius
				delta = np.hstack([delta, np.array([0])]) # make delta's dimention equal to temp[index]'s dimention
				temp[index] = p + delta
				index += 1

		spare_pos = np.random.random_sample((spare, self._dimention+1))# spare particle will randomly spawn without consider about weight
		bound = np.vstack([self._boundary,np.array([0,0])])
		spare_pos = (bound[:,1] - bound[:,0])*spare_pos[:] + bound[:,0]
		temp[-spare:] = spare_pos[:]
		temp[:,-1] = 0
		self._flock = temp

	def move_particles(self, translation):
		if len(translation) != self._dimention:
			raise ValueError("translation must be array like with dimention (dimention, 1)")
		self._flock[:,:-1] += translation

	def __getitem__(self, index):
		return self.Particle(self._flock[index])

	class Particle(object):

		def __init__(self, array_data):
			self.__data = array_data

		@property
		def position(self):
			return self.__data[:-1]

		@property
		def weight(self):
			return self.__data[-1]

		@property
		def dimention(self):
			return len(self.__data)-1

	@property
	def n_particle(self):
		return self._n_particle

	@property
	def dimention(self):
		return self._dimention

	@property
	def average(self):
		return np.average(self._flock, axis=0, weights = self._flock[:,-1]) if sum(self._flock[:,-1]) > 0 else np.average(self._flock, axis=0)

	@property
	def max_weight(self):
		return self._flock[:,-1].max()

class Map1D(object):

	def __init__(self,max_w = 640, max_h = 640, pos = 0):
		self._mountain_hieght = np.zeros([max_w], dtype = int)
		self._max_w = max_w
		self._max_h = max_h
		self._pos = pos
		self.initial_mountain()

	def initial_mountain(self, step = 30):
		np.random.seed()
		rand_num = np.random.randint(0, high = self._max_h/2,  size=(self._max_w/step)+1)
		for i in range(0,self._max_w - step, step):
			for j in range(step):
				self._mountain_hieght[i+j] = ((step-j)*rand_num[(i/step)+1] + j*rand_num[(i/step)])/step

	def measure_hieght(self, position):
		return self._mountain_hieght[int(position)]

	def cost_function(self, array):
		if array[0] >= self._max_w or array[0] < 0:
			return 0
		# print self.measure_hieght(array[0]) - self.measure_hieght(self._pos)
		return np.exp(-(self.measure_hieght(array[0]) - self.measure_hieght(self._pos))**2)

	def visualize(self,img):
		for m in range(self._max_w):
			img[ self._max_h - self._mountain_hieght[m] :self._max_h, m] = [0, 255, 100]

	def set_current_pos(self, pos):
		self._pos = pos

def visualize_particles(img, flock):
	average = flock.average
	cv2.cvtColor(img, cv2.COLOR_BGR2HSV, img)
	max_ = flock.max_weight if flock.max_weight > 0.0 else 1.0
	for i in flock:
		try:
			color = int((i.weight/max_)*150)
		except Exception as e:
			color = 1
		cv2.circle(img, (int(i.position[0]), 150), 2, [color,255,255], -1)
	cv2.cvtColor(img, cv2.COLOR_HSV2BGR, img)
	cv2.circle(img, (int(average[0]), 200), 10, [100,0,255], -1)

def visualize_airplane(img, position):
	cv2.circle(img, (int(position), 100), 10, [255,100,0], -1)

def create_color_bar(img):
	h,w,_ = img.shape
	cv2.cvtColor(img, cv2.COLOR_BGR2HSV, img)
	for i in range(0,w):
		color = int((float(i)/float(w))*150)
		img[:25, i, :] = [color, 255, 255]
	cv2.cvtColor(img, cv2.COLOR_HSV2BGR, img)

if __name__ == "__main__":
	img = np.zeros([640, 640,3], dtype = np.uint8)
	create_color_bar(img)
	Map = Map1D()
	Map.visualize(img)
	flock = Particle_Flock(n_particle = 200, dimention = 1, boundary = np.array([[0,640]]), cost_function = Map.cost_function)
	flock.release_particles()

	np.random.seed()
	character = np.random.randint(0, 639) 
	Map.set_current_pos(character)

	random.seed()
	while 1==1:
		img_ = img.copy()
		flock.calculate_cost()
		Map.visualize(img_)
		visualize_particles(img_, flock)
		visualize_airplane(img_, character)

		cv2.imshow('img', img_)
		if cv2.waitKey(0)&0xFF == 27:
			break

		flock.resample(radius = [15], spare_percentage = 5)
		img_ = img.copy()
		flock.calculate_cost()
		Map.visualize(img_)
		visualize_particles(img_, flock)
		visualize_airplane(img_, character)

		cv2.imshow('img', img_)
		if cv2.waitKey(0)&0xFF == 27:
			break

		move = np.random.randint(-40, 40)
		scale = 0.15*move if move > 0 else -0.1*move
		scale = 0.001 if scale == 0 else scale
		real_move = np.random.normal(move, scale)
		if character + real_move >= 640 or character+real_move < 0:
			move = 0
			real_move = 0
		character += real_move
		flock.move_particles([move])
		flock.calculate_cost()
		Map.set_current_pos(character)
	
	cv2.destroyAllWindows()