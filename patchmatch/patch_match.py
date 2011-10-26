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
	#print first_patch 
	#print second_patch
	
	diff = first_patch - second_patch
	mse = multiply(diff , diff)
	#print mse , '\n'
	return mse.sum()

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
	for j in range(5):
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

				#print second_image_offset[0] , second_image_offset[1]
				second_x_from = second_image_offset[0] - x_patch_offset
				second_x_to = second_image_offset[0] + (x_patch_offset + 1)

				second_y_from = second_image_offset[1] - y_patch_offset
				second_y_to = second_image_offset[1] + (y_patch_offset + 1)

				second_image_patch = second_image[second_x_from:second_x_to , second_y_from:second_y_to]
				patch_difference = difference_function(first_image_patch , second_image_patch)

				#print 'Current patch difference ' , patch_difference

				"""if patch_difference < patch_difference_threshold:
					#Propogation. This isn't exactly how it was described in the paper, but it should be pretty close?
					#Could check here if their current difference is less than what we would be replacing it with
					nnf[x - 1 , y] = second_image_offset
					nnf[x , y - 1] = second_image_offset
				else:
					nnf[x , y , 0] = random_integers(patch_size[0] , nnf_size[0] - patch_size[0] , nnf_size)
					nnf[x , y , 1] = random_integers(patch_size[1] , nnf_size[1] - patch_size[1] , nnf_size)
				"""

				if patch_difference < 10:
					continue
					

				previous_x_patch_difference = patch_difference 
				previous_y_patch_difference = patch_difference

				#First, we check if the pixels above or below us have a better match than us.
				if x > 0 and y > 0:
					previous_x_offset = nnf[x - 1 , y]
					previous_y_offset = nnf[x , y - 1]

					if previous_x_offset[0] + x_patch_offset + 1 < size(second_image , axis=0):
						previous_x_from = previous_x_offset[0] - x_patch_offset + 1
						previous_x_to = previous_x_offset[0] + (x_patch_offset + 1) + 1

						previous_y_from = previous_x_offset[1] - y_patch_offset 
						previous_y_to = previous_x_offset[1] + (y_patch_offset + 1)

						previous_x_patch = second_image[previous_x_from:previous_x_to , previous_y_from:previous_y_to]

						#print previous_x_patch
						#print first_image_patch , '\n'

						previous_x_patch_difference = difference_function(first_image_patch , previous_x_patch)


					if previous_y_offset[1] + y_patch_offset + 1 < size(second_image , axis=1):
						previous_x_from = previous_y_offset[0] - x_patch_offset
						previous_x_to = previous_y_offset[0] + (x_patch_offset + 1)

						previous_y_from = previous_y_offset[1] - y_patch_offset + 1
						previous_y_to = previous_y_offset[1] + (y_patch_offset + 1) + 1

						previous_y_patch = second_image[previous_x_from:previous_x_to , previous_y_from:previous_y_to]
						
						#print previous_y_patch
						#print first_image_patch , '\n'

						previous_y_patch_difference = difference_function(first_image_patch , previous_y_patch)

					if previous_x_patch_difference < patch_difference:
						second_image_offset = previous_x_offset
						patch_difference = previous_x_patch_difference

					if previous_y_patch_difference < patch_difference:
						second_image_offset = previous_y_offset
						patch_difference = previous_y_patch_difference

				#print 'Patch difference after propogation ' , patch_difference
										
				#Set our nnf to the new min value (Could just be the old value)
				#nnf[x , y] = second_image_offset
				#Now we are done propogating, or not, so we do a random search of our area for better matches

				i = 0
				a = .5
	
				search_radius = second_image.shape[0] * pow(a , i) 				
				#search_radius = 0

				while search_radius > 1:
					x_search_offset = round(second_image_offset[0] + search_radius * uniform(-1,1))
					y_search_offset = round(second_image_offset[1] + search_radius * uniform(-1,1))

					if x_search_offset < x_patch_offset:
						x_search_offset = x_patch_offset
					
					if x_search_offset > second_image.shape[0] - (x_patch_offset + 1):
						x_search_offset = second_image.shape[0] - (x_patch_offset + 1)

					if y_search_offset < y_patch_offset:
						y_search_offset = y_patch_offset

					if y_search_offset > second_image.shape[1] - (y_patch_offset + 1):
						y_search_offset = second_image.shape[1] - (y_patch_offset + 1)	
				
					x_search_from = x_search_offset - x_patch_offset
					x_search_to = x_search_offset + x_patch_offset + 1

					y_search_from = y_search_offset - y_patch_offset
					y_search_to = y_search_offset + y_patch_offset + 1

					#print x_search_from , x_search_to
					#print y_search_from , y_search_to

					second_image_patch = second_image[x_search_from:x_search_to , y_search_from:y_search_to]

					search_patch_difference = difference_function(first_image_patch , second_image_patch)
					
					if search_patch_difference < patch_difference:
						second_image_offset = [x_search_offset , y_search_offset]
						patch_difference = search_patch_difference

					i += 1
					
					search_radius = second_image.shape[0] * pow(a , i) 				

					
				#print 'Patch difference after random search ' , patch_difference , '\n'
				#Now we are done propogating and iterating, we store the lowest patch offset and move on
				nnf[x , y] = second_image_offset
		print j

	return nnf

def shift_filter(input):
	#input = reshape(input , (5 , 5))
	#shift = [[0 , 0 , 0 , 0 , 0] , [0 , 0 , 0 , 0 , 0] , [0 , 0 , 0 , 0 , 0] , [0 , 0 , 0 , 0 , 0] , [0 , 0 , 0 , 0 , 1]]
	#return input * shift
	return input[-1]

def apply_nnf(nnf , original):
	shifted = zeros(original.shape)

	for y in range(original.shape[1]):
		for x in range(original.shape[0]):
			shift_x = nnf[x , y, 0]
			shift_y = nnf[x , y , 1]
				

			shifted[x , y] = original[shift_x , shift_y]
	
	return shifted

lena_un = lena()

lena_shifted = generic_filter(lena_un , shift_filter , (101 , 101) , mode='wrap')
print lena_shifted

nnf = nearest_neighbor_field(lena_shifted , lena_un , patch_size=(1,1))

print nnf

fixed_sh = apply_nnf(nnf , lena_un)

diff_sh = fixed_sh - lena_un

imsave('lena_un.tif' , lena_un)
imsave('lena_sh.tif' , lena_shifted)
imsave('fixed_sh.tif' , fixed_sh)
imsave('diff_sh.tif' , diff_sh)

lena_hole = lena()
lena_hole[100:200 , 100:200] = 0

nnf = nearest_neighbor_field(lena_un , lena_hole , patch_size=(7,7))

print nnf

fixed_hole = apply_nnf(nnf , lena_un)

diff_hole = fixed_hole - lena_un

imsave('lena_hole.tif' , lena_hole)
imsave('fixed_hole.tif' , fixed_hole)
imsave('diff_hole.tif' , diff_hole)
