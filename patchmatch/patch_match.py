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
	#print 'First ' , first_patch 
	#print 'Second ' , second_patch

	first_patch_32 = int32(first_patch)
	second_patch_32 = int32(second_patch)

	diff = subtract(first_patch_32 , second_patch_32)
	#print 'Diff ' , diff
	mse = multiply(diff , diff)
	#print 'MSE ' , mse, '\n'
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
	
	#print nnf

	return nnf
def update_nearest_neighbor_field(first_image , second_image , nnf , difference_function , patch_size):
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

			if patch_difference < 4 * len(patch_size):
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

					previous_x_patch_difference = difference_function(first_image_patch , previous_x_patch)


				if previous_y_offset[1] + y_patch_offset + 1 < size(second_image , axis=1):
					previous_x_from = previous_y_offset[0] - x_patch_offset
					previous_x_to = previous_y_offset[0] + (x_patch_offset + 1)

					previous_y_from = previous_y_offset[1] - y_patch_offset + 1
					previous_y_to = previous_y_offset[1] + (y_patch_offset + 1) + 1

					previous_y_patch = second_image[previous_x_from:previous_x_to , previous_y_from:previous_y_to]
						
					previous_y_patch_difference = difference_function(first_image_patch , previous_y_patch)

				if previous_x_patch_difference < patch_difference:
					second_image_offset = [previous_x_offset[0] + 1, previous_x_offset[1]]
					patch_difference = previous_x_patch_difference

				if previous_y_patch_difference < patch_difference:
					second_image_offset = [previous_y_offset[0] , previous_y_offset[1] + 1]
					patch_difference = previous_y_patch_difference

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

	return nnf

def update_image(nnf , image ,  patch_size):
	updated_image = image

	for y in range(0 , image.shape[1]):
		for x in range(0 , image.shape[0]):
			x_offset = int(floor(patch_size[0] / 2))

			x_from = -1 * x_offset
			x_to = x_offset + 1


			y_offset = int(floor(patch_size[1] / 2))
			y_from = -1 * y_offset
			y_to = y_offset + 1

			image_average = 0

			for i in range(x_from , x_to):
				for j in range(y_from , y_to):
					x_index = x + i

					if x_index < 0:
						x_index = 0

					if x_index > image.shape[0] - 1:
						x_index = image.shape[0] - 1

					y_index = y + j

					if y_index < 0:
						y_index = 0

					if y_index > image.shape[1] - 1:
						y_index = image.shape[1] - 1
					

					image_offset = nnf[x_index , y_index]

					image_value = image[image_offset[0] , image_offset[1]]
				
					image_average += image_value

			image_average /= patch_size[0] * patch_size[1]
			
			nnf_offset = nnf[x , y]
			#print image[nnf_offset[0] , nnf_offset[1]] , image_average

			updated_image[x , y] = image_average

	return updated_image

def nearest_neighbor_field(first_image , second_image , difference_function=default_patch_difference , patch_size=(1 , 1)):
	patch_difference_threshold = 10	
	
	nnf = initialize_nearest_neighbor_field(first_image.shape , patch_size)
	
	print patch_size
	


	#second_image = imresize(second_image , second_image.shape[0] / 8 , second_image.shape[1] / 8)

	#for i in (1 , 2):
	#	second_image = imresize(second_image , ((second_image.shape[0] / 2) * i , (second_image.shape[1] / 2) * i))

	for j in range(5):
		nnf = update_nearest_neighbor_field(first_image , second_image , nnf , difference_function , patch_size)
			
		nnf_rgb = zeros((nnf.shape[0] , nnf.shape[1] , 3))
		nnf_rgb[: , : , 0] = nnf[: , : , 0]
		nnf_rgb[: , : , 1] = nnf[: , : , 1]

		imsave('nnf_' + str(j) + '_' + '.tif' , nnf_rgb)
		

		second_image = update_image(nnf , second_image , patch_size)

		imsave('fixed_' + str(j) + '.tif' , second_image)

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

im_one = imread('camel.jpg' , flatten=1)
im_two = imread('camel.jpg' , flatten=1)

#im_one = imresize(im_one , (im_one.shape[0] / 8 , im_one.shape[1] / 8))
#im_two = imresize(im_two , (im_two.shape[0] / 8 , im_two.shape[1] / 8))
im_two[100:200 , 300:450] = 255

#im_one = imresize(im_one , (512 , 512))
#im_two = imresize(im_two , (512 , 512))

imsave('camel_1.tif' , im_one)
imsave('camel_2.tif' , im_two)


print im_one.shape
print im_two.shape


nnf = nearest_neighbor_field(im_one , im_two , patch_size=(3 , 3))

fixed = apply_nnf(nnf , im_two)

imsave('camel_fixed.tif' , fixed)
"""
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
"""
