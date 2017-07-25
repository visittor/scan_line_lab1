import cv2
import numpy as np 

def sum_image(int_img, x, y, dimention):
	sum_ = int_img[y+dimention[0]-1, x+dimention[1]-1] 
	size = int_img.shape
	if size[1] - x < dimention[1] - 1:
		dimention[1] = size[1] - x + 1
	if size[0] - y < dimention[0] - 1:
		dimention[0] = size[0] - y + 1
	if x != 0:
		sum_ -= int_img[y+dimention[0]-1, x-1]
	if y != 0:
		sum_ -= int_img[y-1, x+dimention[1]-1]
	if x != 0 and y != 0:
		sum_ += int_img[y-1, x-1]
	return sum_

def edge_haar_top(int_img, x, y, dimention):
	"""[[ 1  1  1  1],
		[ 1  1  1  1],
		[-1 -1 -1 -1],
		[-1 -1 -1 -1]]"""

	sum_top = sum_image(int_img, x, y, (dimention[0]/2,dimention[1]))
	sum_bot = sum_image(int_img, x, y+(dimention[0])/2, (dimention[0]//2, dimention[1]))
	return sum_top - sum_bot

def edge_haar_bot(int_img, x, y, dimention):
	"""[[ -1 -1 -1 -1],
		[ -1 -1 -1 -1],
	 	[  1  1  1  1],
		[  1  1  1  1]]"""

	sum_top = sum_image(int_img, x, y, (dimention[0]/2,dimention[1]))
	sum_bot = sum_image(int_img, x, y+(dimention[0])/2, (dimention[0]/2, dimention[1]))
	return sum_top - sum_bot

def edge_haar_left(int_img, x, y, dimention):
	"""[[  1  1 -1 -1],
		[  1  1 -1 -1],
		[  1  1 -1 -1],
		[  1  1 -1 -1]]"""

	sum_left = sum_image(int_img, x, y, (dimention[0], dimention[1]/2))
	sum_righ = sum_image(int_img, x+dimention[1]/2, y, (dimention[0],  dimention[1]/2))
	return sum_left - sum_righ

def edge_haar_right(int_img, x, y, dimention):
	"""[[ -1 -1  1  1],
		[ -1 -1  1  1],
		[ -1 -1  1  1],
		[ -1 -1  1  1]]"""

	sum_left = sum_image(int_img, x, y, (dimention[0], dimention[1]/2))
	sum_righ = sum_image(int_img, x+dimention[1]/2, y, (dimention[0],  dimention[1]/2))
	return sum_righ - sum_left

def point_haar_top_left(int_img, x, y, dimention):
 	"""[[  1  1  1  1],
		[  1  1  1  1],
		[  1  1 -1 -1],
		[  1  1 -1 -1]]"""

	sum_all = sum_image(int_img, x, y, dimention)
	sum_piece = sum_image(int_img, x+(dimention[1]/2), y+(dimention[0]/2), (dimention[1]/2, dimention[0]/2))
	return sum_all - sum_piece

def point_haar_top_right(int_img, x, y, dimention):
	"""[[  1  1  1  1],
		[  1  1  1  1],
		[ -1 -1  1  1],
		[ -1 -1  1  1]]"""

	sum_all = sum_image(int_img, x, y, dimention)
	sum_piece = sum_image(int_img, x, y+(dimention[0]/2), (dimention[1]/2, dimention[0]/2))
	return sum_all - sum_piece

def point_haar_bot_left(int_img, x, y, dimention):
	"""[[  1  1 -1 -1],
		[  1  1 -1 -1],
		[  1  1  1  1],
		[  1  1  1  1]]"""

	sum_all = sum_image(int_img, x, y, dimention)
	sum_piece = sum_image(int_img, x+(dimention[1]/2), y, (dimention[1]/2, dimention[0]/2))
	return sum_all - sum_piece

def point_haar_top_left(int_img, x, y, dimention):
	"""[[ -1 -1  1  1],
		[ -1 -1  1  1],
		[  1  1  1  1],
		[  1  1  1  1]]"""

	sum_all = sum_image(int_img, x, y, dimention)
	sum_piece = sum_image(int_img, x, y, (dimention[1]/2, dimention[0]/2))
	return sum_all - sum_piece























