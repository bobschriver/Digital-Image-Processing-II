import math
import time

from scipy import ndimage , misc
from scipy.misc.pilutil import imread , imsave
from scipy.misc import *
from scipy.ndimage.filters import *
from numpy import *


def background_luminance(input):
	
	b = (1.0 / 32.0 ) * array([[1 , 1 , 1 , 1 , 1] , [1 , 2 , 2 , 2 , 1]  , [1 , 2 , 0 , 2 , 1] , [1 , 2 , 2 , 2 , 1] , [1 , 1 , 1 , 1 , 1]] , double)
	return (input * b).sum()

def max_average_luminance(input):
	g1 = (1.0 / 16.0) * array([[0 , 0 , 0 , 0 , 0] , [1 , 3 , 8 , 3 , 1] , [ 0 , 0 , 0 , 0 , 0] , [-1 , -3 , -8 , -3 , -1] , [ 0 , 0 , 0 , 0 , 0]] , double)

	g2 = (1.0 / 16.0) * array([[0 , 0 , 1 , 0 , 0] , [0 , 8 , 3 , 0 , 0] , [1 , 3 , 0 , -1 , -3] , [ 0 , 0 , -3 , -8 , 0] , [0 , 0 , -1 , 0 , 0]] , double)

	g3 = (1.0 / 16.0) * array([[0 , 0 , 1 , 0 , 0] , [0 , 0 , 3 , 8 , 0] , [-1 , -3 , 0 , 3 , 1] , [0 , -8 , -3 , 0 , 0] , [0 , 0 , -1 , 0 , 1]] , double)

	g4 = (1.0 / 16.0) * array([[0 , 1 , 0 , -1 , 0] , [0 , 3 , 0 , -3 , 0] , [0 , 8 , 0 , -8 , 0] , [0 , 3 , 0 , -3 , 0] , [0 , 1 , 0 , -1 , 0]] , double)

	#print "g1 " , g1 , " g2 " , g2 , " g3 " , g3 , " g4 " , g4

	g1_result = (input * g1).sum()
	g2_result = (input * g2).sum()
	g3_result = (input * g3).sum()
	g4_result = (input * g4).sum()

	return max(g1_result , g2_result , g3_result , g4_result)

def luminance_mask( input ):
	#Constants defined in paper
	l = (1 / 2)
	y = (3 / 128)
	t = 17

	input = reshape(input , (5 , 5))

	#need to resize the array first
	bg = background_luminance(input)
	mg = max_average_luminance(input)

	#print "Background Luminance" , bg
	#print "Max Average Luminance" , mg

	#These magic values are in the paper, will have
	#to figure out what they mean
	f1 = mg * (.0001 * bg + .115) + (l - .01 * bg)
	#print "f1" , f1

	f2 = 0

	if bg <= 127:
		f2 = t * ( 1 - sqrt( bg / 127.0 ) ) + 3
	else:
		f2 = y * (bg - 127.0) + 3
	#print "f2" , f2

	return max(f1 , f2)
	

def texture_mask( input ):
	#Needs a 3x3 window
	center_index = floor(input.size / 2)
	center = input[center_index]
	average =  ( 1.0 / input.size) * input.sum()

	return abs(center - average)

def dilation_filter( input ):
	if True in input:
		return True
	else:
		return 0

def edge_mask( image ):
	#not sure if we want this to be a generic filter, or just 
	#apply a couple masks to an image
	#We first apply the laplacian filter
	#laplace_output = laplace(image)
	#print laplace_output

	#The paper says to use the Canny algorithm, but we'll try sobel first
	#sobel_output = sobel(input)
	#print sobel_output
	
	#finally, we apply the dilation filter
	#edge_mask = generic_filter(sobel_output , dilation_filter , (3 , 3))
	#print edge_mask

	#return edge_mask


	grad_x = ndimage.sobel(image, 0) 
	grad_y = ndimage.sobel(image, 1) 
	grad_mag = numpy.sqrt(grad_x**2+grad_y**2) 
	grad_angle = numpy.arctan2(grad_y, grad_x) 
	# next, scale the angles in the range [0, 3] and then round to quantize 
	quantized_angle = numpy.around(3 * (grad_angle + numpy.pi) / (numpy.pi * 2)) 
	
	# Non-maximal suppression: an edge pixel is only good if its magnitude is 
	# greater than its neighbors normal to the edge direction. We quantize 
	# edge direction into four angles, so we only need to look at four 
	# sets of neighbors 

	_NW = array([[1 , 0 , 0] , [0 , 0 , 0] , [0 , 0 , 1]])
	_W = array([[0 , 0 , 0] , [1 , 0 , 1] , [0 , 0 , 0]])
	_NE = array([[0 , 0 , 1] , [0 , 0 , 0] , [1 , 0 , 0]])
	_N = array([[0 , 1 , 0] , [0 , 0 , 0] , [0 , 1 , 0]])


	_NE_d = 0 
	_W_d = 1 
	_NW_d = 2 
	_N_d = 3 
	
	NE = ndimage.maximum_filter(grad_mag, footprint=_NE) 
	W  = ndimage.maximum_filter(grad_mag, footprint=_W) 
	NW = ndimage.maximum_filter(grad_mag, footprint=_NW) 
	N  = ndimage.maximum_filter(grad_mag, footprint=_N) 
	
	thinned = (((grad_mag > W)  & (quantized_angle == _N_d )) | 
		((grad_mag > N)  & (quantized_angle == _W_d )) | 
		((grad_mag > NW) & (quantized_angle == _NE_d)) | 
		((grad_mag > NE) & (quantized_angle == _NW_d)) ) 
	
	thinned_grad = thinned * grad_mag 
	# Now, hysteresis thresholding: find seeds above a high threshold, then 	
	# expand out until we go below the low threshold 
	#print thinned_grad
	
	high = thinned_grad > 64
	low = thinned_grad > 16
	canny_edges = ndimage.binary_dilation(high, structure=numpy.ones((3,3)), iterations=-1, mask=low) 
	print canny_edges

	edge_mask = generic_filter(canny_edges , dilation_filter , (3 , 3))
	print edge_mask

	return edge_mask

im = lena()

edge = edge_mask( im ) * 255
print edge
imsave("edge.jpg" , edge)


texture = generic_filter(im , texture_mask , (3 , 3))
print texture
imsave("texture.jpg" , texture)

luminance = generic_filter(im , luminance_mask , (5 , 5))
print luminance
imsave("luminance.jpg" , luminance)

texture_edge = zeros((im.shape[0] , im.shape[1] , 2))
texture_edge[ : , : , 0] = texture
texture_edge[ : , : , 1] = edge

min_texture_edge = texture_edge.min(axis=2)
print min_texture_edge

luminance_t_e = zeros((im.shape[0] , im.shape[1] , 2))
luminance_t_e[ : , : , 0] = luminance
luminance_t_e[ : , : , 1] = min_texture_edge

hvz_mask = luminance_t_e.max(axis=2)
print hvz_mask

imsave("mask.jpg" , hvz_mask)
