import math
import time
import matplotlib.pyplot as plt

from scipy import ndimage , misc
from scipy.misc.pilutil import imread , imsave
from numpy import *


def bilateral_filter_old(input):
	global call_count
	
	call_count += 1
	
	print call_count
	print time.time() 

	#TODO: Replace these with parameters
	size = (31 , 31)
	
	distance_deviations = 5
	intensity_deviations = 5

	x_size = size[0]
	y_size = size[1]

	#Assumes a odd size
	x_offset = int(math.floor(x_size / 2))
	y_offset = int(math.floor(y_size / 2))

	x_start = x_offset * -1
	x_end = x_offset

	y_start = y_offset * -1
	y_end = y_offset

	filter_value = 0
	normalization_factor = 0

	fixed_origin = x_offset + y_offset * y_size
	#print time.time()

	gauss_function = lambda distance , deviations : math.pow(math.e , (-0.5) * math.pow( distance / deviations , 2))			


	for y in range(y_start , y_end + 1):
		for x in range(x_start , x_end + 1):
			euclidian_distance = math.hypot(x , y)
			
			fix_x = x + x_offset
			fix_y = (y + y_offset) * y_size
			#print fix_x , fix_y

			fixed_index = (fix_x + fix_y) - 1
			#print fixed_index		
			
			intensity_distance = abs(input[fixed_index] - input[fixed_origin])
			#print "Euclidian Distance " ,  euclidian_distance , " Intensity Distance " , intensity_distance
	
			#gauss_function = lambda distance , deviations : math.pow(math.e , (-0.5) * math.pow( distance / deviations , 2))			

			g_euclidian_distance = math.pow(math.e , (-0.5) * math.pow( euclidian_distance / distance_deviations , 2))
			#print "Gauss Euclidian Distance " , g_euclidian_distance

			g_intensity_distance = math.pow(math.e , (-0.5) * math.pow( intensity_distance / intensity_deviations , 2))
			#print "Gauss Intensity Distance " , g_intensity_distance
			
			filter_value += g_euclidian_distance * g_intensity_distance * input[fixed_index]
			normalization_factor += g_euclidian_distance * g_intensity_distance

	print time.time() , '\n'
	#print filter_value , normalization_factor
	return filter_value * ( 1 / normalization_factor)

def create_gaussian_kernel(size , deviations):
	#x, y = mgrid[:size , :size]
    	#g = exp(-(x**2/float(size)+y**2/float(size)))
    	#return g / g.sum()

    	x = arange(0.0, size, 1.0)
 	y = x[:,newaxis]
    	x0 = y0 = size // 2
    	return exp(-4*log(2) * ((x-x0)**2 + (y-y0)**2) / deviations**2)

call_count = 0
calls_to_print = [90 , 119 , 123 , 127 , 128 , 132 , 136 , 160]

def bilateral_filter(input , size , distance_deviations , intensity_deviations):
	#print time.time()	
	#distance_deviations = 5
	#intensity_deviations = 5

	#size = (31 , 31)

	input = reshape(input , size)
	
	x_size = size[0]
	y_size = size[1]

	#Assumes a odd size
	x_offset = int(math.floor(x_size / 2))
	y_offset = int(math.floor(y_size / 2))

	origin = (x_offset , y_offset)

	g_distance = create_gaussian_kernel(x_size , distance_deviations)
	
	origin_value = input[origin]
	origin_matrix = zeros(size)
	origin_matrix += origin_value

	#difference_matrix = origin_matrix - input
	#difference_matrix /= intensity_deviations
	#linalg.matrix_power(difference_matrix , 2)
	#difference_matrix = g_distance * difference_matrix
	#print difference_matrix
	
	#print time.time()
	intensity_function = lambda intensity : math.pow(math.e , (-0.5) * math.pow( (intensity - origin_value) / intensity_deviations , 2))
	v_intensity_function = vectorize(intensity_function)
	g_intensity = v_intensity_function(input)
	#print time.time()

	

	filter = g_distance * g_intensity
	
	if call_count in calls_to_print:
		plt.plot(filter)

	total = sum(filter * input)
	normalization = sum(filter)
	#print time.time() , "\n"
	call_count += 1
	return total / normalization
	#return 0
	
#Create the image array
size = 128
input = np.zeros((size , size))
input += 128
input[size / 2:size , :size / 2] += 64

1dinput = input[128,] 
filter_size = 31
filter = (filter_size , 1)

distance_deviation = 5
intensity_deviation = 50

gaussian_result = ndimage.filters.gaussian_filter(1dinput , 5 , filter_size)
median_result = ndimage.filters.median_filter(1dinput , filter_size)
bilateral_result = ndimage.filters.generic_filter(1dinput , bilateral_filter , filter , extra_arguments=(filter , distance_deviation , intensity_deviation))

plt.plot(1dinput , gaussian_result , median_result , bilateral_result)

#random = random.standard_normal((size , size))
#input += random 

"""
input = imread("lena-128x128.jpg" , flatten=1)

distance_deviations = [10]
intensity_deviations = [50 , 100 , 300]


for distance_deviation in distance_deviations:
	for intensity_deviation in intensity_deviations:
	
		filter_size = 6 * distance_deviation + 1 
		filter = (filter_size , filter_size)	

		print time.time()
		output = ndimage.filters.generic_filter(input , bilateral_filter , filter , extra_arguments=(filter , distance_deviation , intensity_deviation))
		print time.time()
		filename = "lena-128x128-" + str(distance_deviation) + "-" + str(intensity_deviation) + ".jpg"
		print "Writing to " , filename , "\n"
		imsave(filename , output)	
"""
