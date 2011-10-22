import math
import time

from scipy import ndimage , misc
from scipy.misc.pilutil import imread , imsave
from scipy.misc import *
#from scipy.signal import *
from scipy.ndimage.filters import *
from numpy import *
from numpy.random import *
from numpy.linalg import *



def default_patch_difference(first_patch , second_patch):
	print first_patch , second_patch
	return 1

def initialize_nearest_neighbor_field(nnf_size , patch_size):
	#nnf = ones(nnf_size)
	nnf = zeros((nnf_size[0] , nnf_size[1] , 2))

	#print random_integers(patch_size[0] , nnf_size[0] - patch_size[0] , nnf.shape)
	nnf[: , : , 0] = random_integers(patch_size[0] , nnf_size[0] - patch_size[0] , nnf_size)
	nnf[: , : , 1] = random_integers(patch_size[1] , nnf_size[1] - patch_size[1] , nnf_size)

	#for i in range(nnf_size[0]):
	#	for j in range(nnf_size[1]):
	#		nnf[i , j , : , :] = ones(patch_size) * random_integers(0 , nnf_size[0] * nnf_size[1])
	
	print nnf

	return nnf

def nearest_neighbor_field(first_image , second_image , difference_function=default_patch_difference , patch_size=(1 , 1)):
	patch_difference_threshold = 10	
	
	nnf = initialize_nearest_neighbor_field(first_image.shape , patch_size)
	
	print patch_size
	for i in range(10):
		for y in range(patch_size[1]  , first_image.shape[1] - patch_size[1]):
			for x in range(patch_size[0] , first_image.shape[0] - patch_size[0]):
				x_patch_offset = floor(patch_size[0] / 2)
				
				first_x_from = x - x_patch_offset
				first_x_to = x + x_patch_offset + 1

				y_patch_offset = floor(patch_size[1] / 2)

				first_y_from = y - y_patch_offset
				first_y_to = y + y_patch_offset + 1

				first_image_patch = first_image[first_x_from:first_x_to , first_y_from:first_y_to]
							
				second_image_offset = nnf[x , y]

				print second_image_offset[0] , second_image_offset[1]
				
				second_image_patch = second_image[(second_image_offset[0] , second_image_offset[1])]
				patch_difference = difference_function(first_image_patch , second_image_patch)

				if patch_difference < patch_difference_threshold:
					nnf[x - 1 , y] = second_image_offset
					nnf[x , y - 1] = second_image_offset
				else:
					nnf[x , y , 0] = random_integers(patch_size[0] , nnf_size[0] - patch_size[0] , nnf_size)
					nnf[x , y , 1] = random_integers(patch_size[1] , nnf_size[1] - patch_size[1] , nnf_size)

				
nearest_neighbor_field(zeros((256 , 256)) , zeros((256 , 256)))
