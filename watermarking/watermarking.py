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
		f2 = t * ( 1 - sqrt( bg / 127.0 ) ) + 5
	else:
		f2 = y * (bg - 127.0) + 10
	#print "f2" , f2

	return max(f1 , f2)
"""

def max_averave_luminance( image ):
	b = (1.0 / 32.0 ) * array([[1 , 1 , 1 , 1 , 1] , [1 , 2 , 2 , 2 , 1]  , [1 , 2 , 0 , 2 , 1] , [1 , 2 , 2 , 2 , 1] , [1 , 1 , 1 , 1 , 1]] , double)
	return convolve(image , b)

def background_luminance( image ):
	g1 = (1.0 / 16.0) * array([[0 , 0 , 0 , 0 , 0] , [1 , 3 , 8 , 3 , 1] , [ 0 , 0 , 0 , 0 , 0] , [-1 , -3 , -8 , -3 , -1] , [ 0 , 0 , 0 , 0 , 0]] , double)

	g2 = (1.0 / 16.0) * array([[0 , 0 , 1 , 0 , 0] , [0 , 8 , 3 , 0 , 0] , [1 , 3 , 0 , -1 , -3] , [ 0 , 0 , -3 , -8 , 0] , [0 , 0 , -1 , 0 , 0]] , double)

	g3 = (1.0 / 16.0) * array([[0 , 0 , 1 , 0 , 0] , [0 , 0 , 3 , 8 , 0] , [-1 , -3 , 0 , 3 , 1] , [0 , -8 , -3 , 0 , 0] , [0 , 0 , -1 , 0 , 1]] , double)

	g4 = (1.0 / 16.0) * array([[0 , 1 , 0 , -1 , 0] , [0 , 3 , 0 , -3 , 0] , [0 , 8 , 0 , -8 , 0] , [0 , 3 , 0 , -3 , 0] , [0 , 1 , 0 , -1 , 0]] , double)

	#print "g1 " , g1 , " g2 " , g2 , " g3 " , g3 , " g4 " , g4

	g1_result = convolve(input , g1)
	g2_result = convolve(input , g2)
	g3_result = convolve(input , g3)
	g4_result = convolve(input , g4)
	
	g_sum = zeros((image.shape[0] , input.shape[1] , 4))

	g_sum[ : , : , 0] = g1_result
	g_sum[ : , : , 1] = g2_result
	g_sum[ : , : , 2] = g3_result
	g_sum[ : , : , 3] = g4_result

	return g_sum.max(axis=2)


def luminance_mask( image ):

	bg = background_luminance( image )
	mg = max_average_luminance( image )

	print select([bg < 128] , [bg])
"""

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

def hvs_mask(edge , texture , luminance):
	sum_size = (edge.shape[0] , edge.shape[1] , 2)

	texture_edge_sum = zeros(sum_size)
	texture_edge_sum[ : , : , 0] = texture
	texture_edge_sum[ : , : , 1] = edge

	texture_edge = texture_edge_sum.min(axis=2)

	luminance_te_sum = zeros(sum_size)
	luminance_te_sum[ : , : , 0] = texture_edge * .5
	luminance_te_sum[ : , : , 1] = luminance

	hvz_mask = luminance_te_sum.max(axis=2)
	return hvz_mask

def gen_hvs_mask( image ):
	edge = edge_mask( im ) * 255
	print edge
	imsave("edge.tif" , edge)


	texture = generic_filter(im , texture_mask , (3 , 3))
	print texture
	imsave("texture.tif" , texture)


	#luminance = zeros(im.shape)
	luminance = generic_filter(im , luminance_mask , (5 , 5))
	#luminance = imread("luminance.tif")
	#luminance = luminance_mask(im)
	print luminance
	imsave("luminance.tif" , luminance)

	hvs = hvs_mask(edge , texture , luminance)
	print hvs

	imsave("hvs.tif" , hvs)

	return hvs


#need to add a message here
def watermark(im , hvs_mask , message):
	i = reshape(im , (4 , 4 , -1))
	
	#We have to reshape twice to get the minimum of 
	#the 4x4 block
	t_4_4 = reshape(hvs_mask , (4 , 4 , -1))
	t_16_1 = reshape(t_4_4 , (16 , -1))
	print t_16_1.shape

	min_t = t_16_1.min(axis=0)

	print min_t
	print min_t.shape

	#Generate our watermark
	watermark = random_integers(0 , 1 , (4 , 4))
	watermark[watermark == 0] = -1
	#watermark = ones((4 , 4))
	print watermark

	#Check the message bit here?
	#I think the message is the actual watermark
	#We are embedding

	a = zeros(im.shape)
	print a.shape
	
	#message.ravel()
	#print "message size" , message.size
	#message = message.flatten()
	#print "message" , message
	#print "message shape" , message.shape
	#print "message size" , message.size

	#print "i shape" , i.shape[2]
	#print message[1 , 1]

	#m = 0

	#for j in range(i.shape[2]):
	for y in range((im.shape[1] / 4)):
		for x in range((im.shape[0] / 4)):
			x_from = x * 4
			x_to = (x + 1) * 4
			
			y_from = y * 4
			y_to = (y + 1) * 4

			im_block = im[x_from:x_to , y_from:y_to]
			mask_block = hvs_mask[x_from:x_to , y_from:y_to]
		
			mask_min = mask_block.min()
				
			#print "x " , x , " y " , y
		
			#print "im block" , im_block
			#print "mask_block" , mask_block

			#print "message" , message[x , y]

			if message[x , y] == 1:
				#a[x_from:x_to , y_from:y_to ] = multiply( ( around ( divide(im[x_from:x_to , y_from:y_to] , t_4_4[: , : , j])) + .25 * watermark) , t_4_4[: , : , j])
				a[x_from:x_to , y_from:y_to] = ( around ( divide( im_block ,  mask_block ) ) + .25 * watermark ) * mask_block
				#a[x_from:x_to , y_from:y_to] = 0
			else:
				#a[ : , : , j] = multiply( ( around ( divide(i[: , : , j] , t_4_4[: , : , j])) - .25 * watermark) , t_4_4[: , : , j])
				a[x_from:x_to , y_from:y_to] = ( around ( divide( im_block ,  mask_block ) ) - .25 * watermark ) * mask_block

			#print "a " , a[x_from:x_to , y_from:y_to]
			#print "diff " , a[x_from:x_to , y_from:y_to] - im_block

		#a[: , : , j] += j / 128
		
		#print "i" , i[: , : , j]
		#print "t" , t_4_4[: , : , j]
		#print "w" , watermark
		#print "a" , a[ : , : , j]
		#print "m" , message[j]
		
		#print "diff" , a[: , : , j] - i[: , : , j]
	print a.shape
	
	return (a , watermark)

def gen_message(shape):
	message = zeros((shape[0] / 4, shape[1] / 4))

	#message[64:92 , 64:92] = 1
	#message[10:30 , 25:30] = 1
	#message[35:40 , 10:15] = 1
	#message[200:230 , 200:230] = 1

	message[0:16 , 0:16] = 1
	message[112:128 ,112:128] = 1
	message[16:48 , 16:32] = 1
	message[48:64 , 32:64] = 1
	message[64:96 , 64:80] = 1
	message[96:112 , 80:112] = 1
	
	#message[16:32 , 16:32] = 1

	#message[48:64 , 16:32] = 1

	#message[80:96 , 48:64] = 1

	#message[64:128 , 0:128] = 1

	return message

def decode( im , hvs_mask , watermark):
	#r = reshape(image , (4 , 4 , -1))
	


	#hvs_mask = gen_hvs_mask(im)
	
	#We have to reshape twice to get the minimum of 
	#the 4x4 block
	t_4_4 = reshape(hvs_mask , (4 , 4 , -1))
	t_16_1 = reshape(t_4_4 , (16 , -1))

	min_t = t_16_1.min(axis=0)

	#message = zeros(r.shape[2])

	message = zeros((im.shape[0] / 4 , im.shape[1] / 4))

	#for j in range(r.shape[2]):
	for y in range(im.shape[1] / 4):
		for x in range(im.shape[0] / 4):
			x_from = x * 4
			x_to = (x + 1) * 4
			
			y_from = y * 4
			y_to = (y + 1) * 4
		
			im_block = im[x_from:x_to , y_from:y_to]
			mask_block = hvs_mask[x_from:x_to , y_from:y_to]

			mask_min = mask_block.min()

			#s =  around( divide(im_block , mask_block) ) - divide(im_block , mask_block)
			s = divide(im_block , mask_block) - around( divide( im_block , mask_block ) )
			#print "s " , s
			s = sign( s )
			#print divide(im_block , mask_min)

			#print "s " , s
			#print "w " , watermark

			c = s == watermark
			if (any(all(c , axis=0)) and all(any(c , axis=1))) and (any(all(c , axis=1)) and all(any(c , axis=0))):
			#if all(c):
				message[x , y] = 1
			else:
				message[x , y] = 0

	print message
	return message

def psnr(calc , orig):
	diff = orig - calc
	print diff
	square_diff = multiply(diff , diff)

	print "square " , square_diff

	print "sum" , square_diff.sum() 
	mse = square_diff.sum() / square_diff.size
	
	print "mse " , mse

	psnr = 20 * log10( 1 / sqrt(mse))

	return psnr
	
im = lena()

hvs = gen_hvs_mask(im)

message = gen_message(im.shape)
imsave("message.tif" , message)

(a , watermark) = watermark(im , hvs , message)

imsave("watermarked.tif" , a)

diff = im - a

imsave("wdiff.tif" , diff)

#noise = a
#noise = a + 10
#n = reshape( normal( 0 , 2 , a.size ) , a.shape)
#noise = a + n
#noise = gaussian_filter(a , .5)
noise = (a - 0) / (240 - 0) * 255
#a[0:63, :] = 0
#a[447:511 , :] = 0
#a[ : , 0:63] = 0
#a[ : , 447:511] = 0
#noise = a

print noise

imsave("noisy.tif" , noise)
#noise = add_noise( a )

diff = im - noise

imsave("ndiff.tif" , diff)

decoded = decode(noise , hvs , watermark)
imsave("decode.tif" , decoded)

print psnr(message , decoded)
