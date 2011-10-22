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
		#TODO: change these to maxint
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

				"""if patch_difference < patch_difference_threshold:
					#Propogation. This isn't exactly how it was described in the paper, but it should be pretty close?
					#Could check here if their current difference is less than what we would be replacing it with
					nnf[x - 1 , y] = second_image_offset
					nnf[x , y - 1] = second_image_offset
				else:
					nnf[x , y , 0] = random_integers(patch_size[0] , nnf_size[0] - patch_size[0] , nnf_size)
					nnf[x , y , 1] = random_integers(patch_size[1] , nnf_size[1] - patch_size[1] , nnf_size)
				"""

				#Need to actually calculate this
				previous_x_patch_difference = 100
				previous_y_patch_difference = 100

				#First, we check if the pixels above or below us have a better match than us.
				if x > 0 and y > 0:
					if previous_x_patch_difference < patch_difference:
						second_image_offset = nnf[x - 1 , y]
						#Should check here to make sure we don't go off the ends of the image
						second_image_patch = second_image[(second_image_offset[0] + 1 , second_image_offset[1])]

						patch_difference = difference_function(first_image_patch , second_image_patch)
					
					if previous_y_patch_difference < patch_difference:
						second_image_offset = nnf[x , y - 1]
						second_image_patch = second_image[(second_image_offset[0] , second_image_offset[1] + 1)]
						
						patch_difference = difference_function(first_image_patch , second_image_patch)

				#Set our nnf to the new min value (Could just be the old value)
				#nnf[x , y] = second_image_offset
				#Now we are done propogating, or not, so we do a random search of our area for better matches

				i = 0
				a = .5
	
				search_radius = pow(a , i) 				

				while search_radius > 1:
					x_search_offset = second_image_offset[0] + search_radius * uinform(1)
					y_search_offset = second_image_offset[1] + search_radius * uinform(1)

					if x_search_offset < 0:
						x_search_offset = 0
					
					if x_search_offset > second_image.shape[0]:
						x_search_offset = second_image.shape[0]

					if y_search_offset < 0:
						y_search_offset = 0

					if y_search_offset > second_image.shape[1]:
						y_search_offset = second_image.shape[1]	
					
					second_image_patch = second_image[(x_search_offset , y_search_offset)]

					search_patch_difference = difference_function(first_image_patch , second_image_patch)

					if search_patch_difference < patch_difference:
						second_image_offset = [x_search_offset , y_search_offset]
						patch_difference = search_patch_difference

					i++

				#Now we are done propogating and iterating, we store the lowest patch offset and move on
				nnf[x , y] = second_image_offset
				

nearest_neighbor_field(zeros((256 , 256)) , zeros((256 , 256)))
