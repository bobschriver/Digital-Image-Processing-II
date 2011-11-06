import math
import time
import sys

from scipy import ndimage , misc
from scipy.misc.pilutil import imread , imsave
from scipy.misc import *
#from scipy.signal import *
from scipy.ndimage.filters import *
from numpy import *
from numpy.random import *
from numpy.linalg import *



def default_patch_difference(first_patch , second_patch , first_patch_weight , second_patch_weight):
	#print 'First ' , first_patch.min()
	#print 'Second ' , second_patch.min()

	first_patch_32 = int32(first_patch)
	second_patch_32 = int32(second_patch)

	diff = subtract(first_patch_32 , second_patch_32)
	#print 'Diff ' , diff
	mse = multiply(diff , diff)
	#print 'MSE ' , mse, '\n'
	return mse.sum()

def fill_patch_difference(first_patch , second_patch , first_patch_weight , second_patch_weight):
	first_patch_32 = int32(first_patch)
	second_patch_32 = int32(second_patch)
	#print 'First '  , first_patch
	#print 'Second '  , second_patch
	#print 'First weight ' , first_patch_weight
	#print 'Second weight '  , second_patch_weight

	if first_patch_weight.sum() == 0:
		return sys.maxint

	diff = subtract(first_patch_32 , second_patch_32)
	diff = multiply(diff , first_patch_weight)
	#print 'Diff '  , diff
	mse = multiply(diff , diff)
	mse = mse + second_patch_weight
	#print 'MSE ' , mse
	return mse.sum()

	"""
	top_diff = diff.copy()
	top_diff[5: , 5:] = 0
	#top_diff[5: , :] = 0

	bottom_diff = diff.copy()
	bottom_diff[:6 , :6] = 0
	#bottom_diff[:6 , :] = 0

	left_diff = diff.copy()
	left_diff[:6 , 5:] = 0
	#left_diff[: , 5:] = 0

	right_diff = diff.copy()
	right_diff[5: , :6] = 0
	#right_diff[: , :6]  =0

	#print 'Top diff ' , top_diff
	#print 'Bottom diff ' , bottom_diff
	#print 'Left diff '  , left_diff
	#print 'Right diff ' , right_diff

	top_mse = multiply(top_diff , top_diff)
	bottom_mse = multiply(bottom_diff , bottom_diff)
	left_mse = multiply(left_diff , left_diff)
	right_mse = multiply(right_diff , right_diff)

	#print 'Top MSE ' , top_mse
	#print 'Bottom MSE ' , bottom_mse
	#print 'Left MSE '  , left_mse
	#print 'Right MSE ' , right_mse
	
	#print 'Top MSE ' , top_mse.sum() , ' Bottom MSE ' , bottom_mse.sum() , ' Left MSE ' , left_mse.sum() , ' Right MSE ' , right_mse.sum() 

	return min(top_mse.sum() , bottom_mse.sum() , left_mse.sum() , right_mse.sum())
	#return min(top_mse.sum() , bottom_mse.sum())
	#print 'Diff ' , diff
	#mse = multiply(diff , diff)
	#print 'MSE ' , mse
	#return mse.sum()
	"""

def initialize_nearest_neighbor_field(first_image_size , second_image_size , patch_size , default_x_offset , default_y_offset):
	#nnf = ones(nnf_size)

	nnf_size = first_image_size
	nnf = ones((nnf_size[0] , nnf_size[1] , 2))
	print default_x_offset , default_y_offset
	nnf[: , : , 0] = transpose(range(default_x_offset , nnf_size[0] + default_x_offset) * transpose(nnf[: , : , 0]))
	nnf[: , : , 1] = range(default_y_offset , nnf_size[1] + default_y_offset) * nnf[: , : , 1]

	print nnf_size
	#print random_integers(patch_size[0] , nnf_size[0] - patch_size[0] , nnf.shape)
	nnf[patch_size[0]:(nnf_size[0] - patch_size[0]) , patch_size[1]:(nnf_size[1] - patch_size[1]) , 0] = random_integers(patch_size[0] , second_image_size[0] - patch_size[0] , (nnf_size[0] - 2 * patch_size[0] , nnf_size[1] - 2 * patch_size[1]))
	nnf[10:(nnf_size[0] - 10) , 10:(nnf_size[1] - 10) , 1] = random_integers(patch_size[1] , second_image_size[1] - patch_size[1] , (nnf_size[0] - 20 , nnf_size[1] - 20))
	
	print nnf

	x_patch_offset = int(floor(patch_size[0] / 2))
	y_patch_offset = int(floor(patch_size[1] / 2))



	#for i in range(nnf_size[0]):
	#	for j in range(nnf_size[1]):
	#		nnf[i , j , : , :] = ones(patch_size) * random_integers(0 , nnf_size[0] * nnf_size[1])
	
	print nnf

	return nnf
def update_nearest_neighbor_field(first_image , first_image_weights ,  second_image , second_image_weights , nnf , difference_function , patch_size , direction=0):
	
	
	x_patch_offset = int(floor(patch_size[0] / 2))
	y_patch_offset = int(floor(patch_size[1] / 2))

	y_range = range(y_patch_offset , first_image.shape[1] - (y_patch_offset + 1))
	x_range = range(x_patch_offset , first_image.shape[0] - (x_patch_offset + 1))

	if direction == 1:
		y_range.reverse()
		x_range.reverse()


	for y in y_range:
		for x in x_range:
				
			first_x_from = x - x_patch_offset
			first_x_to = x + x_patch_offset + 1

			if first_x_from < 0:
				first_x_from = 0

			if first_x_to > first_image.shape[0] - 1:
				first_x_to = first_image.shape[0] - 1


			first_y_from = y - y_patch_offset
			first_y_to = y + y_patch_offset + 1

			if first_y_from < 0:
				first_y_from = 0

			if first_y_to > first_image.shape[1] - 1:
				first_y_to = first_image.shape[1] - 1
			
			#print first_x_from , first_x_to , first_y_from , first_y_to

			first_image_patch = zeros(patch_size)
			#first_image_patch_values = first_image[first_x_from:first_x_to , first_y_from:first_y_to]
			
			#first_image_patch[:first_image_patch_values.shape[0] , :first_image_patch_values.shape[1]] = first_image_patch_values
			first_image_patch = first_image[first_x_from:first_x_to , first_y_from:first_y_to]

			second_image_offset = nnf[x , y]

			#print second_image_offset[0] , second_image_offset[1]
			second_x_from = second_image_offset[0] - x_patch_offset
			second_x_to = second_image_offset[0] + (x_patch_offset + 1)

			second_y_from = second_image_offset[1] - y_patch_offset
			second_y_to = second_image_offset[1] + (y_patch_offset + 1)

			second_image_patch = second_image[second_x_from:second_x_to , second_y_from:second_y_to]
			
			first_image_patch_weights = first_image_weights[first_x_from:first_x_to , first_y_from:first_y_to]
			second_image_patch_weights = second_image_weights[second_x_from:second_x_to , second_y_from:second_y_to]

			patch_difference = difference_function(first_image_patch , second_image_patch , first_image_patch_weights , second_image_patch_weights)

			#print patch_difference

			#print 'Current patch difference ' , patch_difference


			previous_x_patch_difference = patch_difference 
			previous_y_patch_difference = patch_difference

			#First, we check if the pixels above or below us have a better match than us.
			#This isn't exatly how the paper does it, it should check the previous patch distance rather than 
			#The previous patch offset + 1's distance, and if its good, set the current patch offset to
			#the previous patch offset + 1
			#This difference shouldn't really affect anything though, but we assume that the previous patch was good
			if x > 0 and y > 0:
				previous_x_offset = nnf[x - 1 , y]
				previous_y_offset = nnf[x , y - 1]

				if previous_x_offset[0] + x_patch_offset + 1 < size(second_image , axis=0):
					previous_x_from = previous_x_offset[0] - x_patch_offset + 1
					previous_x_to = previous_x_offset[0] + (x_patch_offset + 1) + 1

					previous_y_from = previous_x_offset[1] - y_patch_offset 
					previous_y_to = previous_x_offset[1] + (y_patch_offset + 1)

					previous_x_patch = second_image[previous_x_from:previous_x_to , previous_y_from:previous_y_to]
					previous_x_patch_weights = second_image_weights[previous_x_from:previous_x_to , previous_y_from:previous_y_to]
					
					previous_x_patch_difference = difference_function(first_image_patch , previous_x_patch , first_image_patch_weights , previous_x_patch_weights)


				if previous_y_offset[1] + y_patch_offset + 1 < size(second_image , axis=1):
					previous_x_from = previous_y_offset[0] - x_patch_offset
					previous_x_to = previous_y_offset[0] + (x_patch_offset + 1)

					previous_y_from = previous_y_offset[1] - y_patch_offset + 1
					previous_y_to = previous_y_offset[1] + (y_patch_offset + 1) + 1

					previous_y_patch = second_image[previous_x_from:previous_x_to , previous_y_from:previous_y_to]
					previous_y_patch_weights = second_image_weights[previous_x_from:previous_x_to , previous_y_from:previous_y_to]

					previous_y_patch_difference = difference_function(first_image_patch , previous_y_patch , first_image_patch_weights , previous_y_patch_weights)

				if previous_x_patch_difference < patch_difference:
					second_image_offset = [previous_x_offset[0] + 1, previous_x_offset[1]]
					patch_difference = previous_x_patch_difference

				if previous_y_patch_difference < patch_difference:
					second_image_offset = [previous_y_offset[0] , previous_y_offset[1] + 1]
					patch_difference = previous_y_patch_difference

			
			#print 'Patch difference after prop '  , patch_difference
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
				second_image_patch_weights = second_image_weights[x_search_from:x_search_to , y_search_from:y_search_to]

				search_patch_difference = patch_difference
				search_patch_difference = difference_function(first_image_patch , second_image_patch , first_image_patch_weights , second_image_patch_weights)
					
				if search_patch_difference < patch_difference:
					second_image_offset = [x_search_offset , y_search_offset]
					patch_difference = search_patch_difference

				i += 1
					
				search_radius = second_image.shape[0] * pow(a , i) 				
	
			#print 'Patch difference after random search ' , patch_difference , '\n'
			
			#Now we are done propogating and iterating, we store the lowest patch offset and move on
			#We should update the weight fields here
		
			
			#print len(first_image_patch_weights[first_image_patch_weights > 0])

			if patch_difference < len(first_image_patch_weights[first_image_patch_weights > 0]) * 10:
				first_image_weights[x , y] = 1

			nnf[x , y] = second_image_offset

	return (nnf , first_image_weights)

def update_image(nnf , first_image , second_image ,  patch_size):
	updated_image = first_image

	for y in range(0 , first_image.shape[1]):
		for x in range(0 , first_image.shape[0]):
			x_offset = int(floor(patch_size[0] / 2))

			x_from = -1 * x_offset
			x_to = x_offset + 1


			y_offset = int(floor(patch_size[1] / 2))
			y_from = -1 * y_offset
			y_to = y_offset + 1
			
			
			image_average = 0
			"""
			for i in range(x_from , x_to):
				for j in range(y_from , y_to):
					x_index = x + i

					if x_index < 0:
						x_index = 0

					if x_index > first_image.shape[0] - 1:
						x_index = first_image.shape[0] - 1

					if y_index < 0:
						y_index = 0

					if y_index > first_image.shape[1] - 1:
						y_index = first_image.shape[1] - 1
					

					image_offset = nnf[x_index , y_index]

					image_value = second_image[image_offset[0] , image_offset[1]]
				
					image_average += image_value

			image_average /= patch_size[0] * patch_size[1]
		"""	
			nnf_offset = nnf[x , y]
			#print image[nnf_offset[0] , nnf_offset[1]] , image_average
			updated_image[x , y] = second_image[nnf_offset[0] , nnf_offset[1]]
			#updated_image[x , y] = image_average

	return updated_image

def nearest_neighbor_field(first_image , first_image_weights , second_image , second_image_weights , nnf , difference_function=default_patch_difference , patch_size=(1 , 1) , resize_steps=0):
	patch_difference_threshold = 10	
	
	
	print patch_size
	
	orig_first_image_shape = first_image.shape
	orig_nnf_shape = nnf.shape

	#second_image = imresize(second_image , second_image.shape[0] / 8 , second_image.shape[1] / 8)

	resize_amounts = (1 , 2 , 4 , 8 , 16)

	test = update_image(nnf , first_image , second_image , patch_size);
	imsave('orig.tif' , test)

	diff = second_image.copy()
	diff[90:210 , 290:462] = diff[90:210 , 290:462] - test
	print diff[90:210 , 290:462]

	imsave('diff.tif' , diff)



	for i in resize_amounts[0:resize_steps+1]:
		#first_image = imresize(first_image , ((orig_first_image_shape[0] / resize_amounts[resize_steps]) * i , (orig_first_image_shape[1] / resize_amounts[resize_steps]) * i))
		#nnf = resize(nnf , ((orig_nnf_shape[0] / resize_amounts[resize_steps]) * i, (orig_nnf_shape[1] / resize_amounts[resize_steps]) * i , 2))
		

		for j in range(7):
			nnf_rgb = zeros((nnf.shape[0] , nnf.shape[1] , 3))
			nnf_rgb[: , : , 0] = nnf[: , : , 0]
			nnf_rgb[: , : , 1] = nnf[: , : , 1]

			
			
			imsave('nnf_' + str(j) + '_' + str(i) + '.tif' , nnf_rgb)
			
			
			(nnf , first_image_weights) = update_nearest_neighbor_field(first_image , first_image_weights , second_image , second_image_weights , nnf , difference_function , patch_size)
			
			imsave('image_weights_' + str(j) + '.tif' , first_image_weights)	
			
			first_image = update_image(nnf ,first_image ,  second_image , patch_size)

			imsave('fixed_' + str(j) + '_' + str(i) + '.tif' , first_image)


			print j
			

	return first_image

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

#im_one = imread('r.jpg' , flatten=1)
im_two = around(imread('camel.jpg' , flatten=1))
print im_two.shape
print im_two
#im_two[125:200 , 325:425] = random_integers(0 , 255 , (75 , 100)) 

#im_two[125:200 , 325:425] = im_two.mean()
im_one = im_two[90:210 , 290:462].copy()

#im_two[125:200 , 325:425] = random_integers(0 , 255 , (75 , 100)) 

im_one_weights = ones(im_one.shape)
im_two_weights = zeros(im_two.shape)

im_one_weights[im_one < 128] = 0 
im_two_weights[im_two < 140] = 128

im_one_weights = floor(gaussian_filter(im_one_weights , .5))
im_two_weights = ceil(gaussian_filter(im_two_weights , .5))

imsave('im_one_weights.tif' , im_one_weights)
imsave('im_two_weights.tif' , im_two_weights)

#nnf = initialize_nearest_neighbor_field(im_one.shape , im_two.shape , (3 , 3) , 90 , 290)

default_x_offset = 90
default_y_offset = 290
nnf_size = im_one.shape
nnf = ones((nnf_size[0] , nnf_size[1] , 2))
print default_x_offset , default_y_offset
print nnf_size
nnf[: , : , 0] = transpose(range(default_x_offset , nnf_size[0] + default_x_offset) * transpose(nnf[: , : , 0]))
nnf[: , : , 1] = range(default_y_offset , nnf_size[1] + default_y_offset) * nnf[: , : , 1]

print nnf
print nnf.shape
ans = nearest_neighbor_field(im_one , im_one_weights , im_two , im_two_weights , nnf , difference_function=fill_patch_difference , patch_size=(11 , 11)) 

fixed = im_two
fixed[90:210 , 290:462] = ans

imsave('camel_fixed.tif' , fixed)


"""
im_one = imresize(im_one , (im_one.shape[0] / 8 , im_one.shape[1] / 8))
im_two = imresize(im_two , (im_two.shape[0] / 8 , im_two.shape[1] / 8))

#im_one = imresize(im_one , (512 , 512))
#im_two = imresize(im_two , (512 , 512))

imsave('river_1_r.tif' , im_one)
imsave('river_2_r.tif' , im_two)


print im_one.shape
print im_two.shape


nnf = initialize_nearest_neighbor_field(im_one.shape , im_two.shape, (1 , 1) , 0 , 0)
ans = nearest_neighbor_field(im_one , im_two , nnf  ,  patch_size=(1 , 1) , resize_steps=0)

imsave('river_fixed.tif' , ans)
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
