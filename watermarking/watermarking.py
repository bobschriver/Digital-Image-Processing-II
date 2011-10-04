def background_luminance(input):
	b = (1 / 32 ) * [[1 , 1 , 1 , 1 , 1] , [1 , 2 , 2 , 2 , 1]  , [1 , 2 , 0 , 2 , 1] , [1 , 2 , 2 , 2 , 1] , [1 , 1 , 1 , 1 , 1]]
	return input * b

def max_average_luminance(input):
	g1 = (1 / 16) * [[0 , 0 , 0 , 0 , 0] , [1 , 3 , 8 , 3 , 1] , [ 0 , 0 , 0 , 0 , 0] , [-1 , -3 , -8 , -3 , -1] , [ 0 , 0 , 0 , 0 , 0]]

	g2 = (1 / 16) * [[0 , 0 , 1 , 0 , 0] , [0 , 8 , 3 , 0 , 0] , [1 , 3 , 0 , -1 , -3] , [ 0 , 0 , -3 , -8 , 0] , [0 , 0 , -1 , 0 , 0]]

	g3 = (1 / 16) * [[0 , 0 , 1 , 0 , 0] , [0 , 0 , 3 , 8 , 0] , [-1 , -3 , 0 , 3 , 1] , [0 , -8 , -3 , 0 , 0] , [0 , 0 , -1 , 0 , 1]]

	g4 = (1 / 16) * [[0 , 1 , 0 , -1 , 0] , [0 , 3 , 0 , -3 , 0] , [0 , 8 , 0 , -8 , 0] , [0 , 3 , 0 , -3 , 0] , [0 , 1 , 0 , -1 , 0]]

	g1_result = input * g1
	g2_result = input * g2
	g3_result = input * g3
	g4_result = input * g4

	return max(g1_result , g2_result , g3_result , g4_result)

def luminance_mask( input ):
	#Constants defined in paper
	l = (1 / 2)
	y = (3 / 128)
	t = 17

	#need to resize the array first
	bg = background_luminance(input)
	mg = max_average_luminance(input)

	#These magic values are in the paper, will have
	#to figure out what they mean
	f1 = mg * (.0001 * bg + .115) + (l - .01 * bg)

	f2 = 0

	if bg <= 127:
		f2 = t * ( 1 - ( ( bg / 127 ) ^ .5 ) ) + 3
	else:
		f2 = y * (bg - 127) + 3

	return max(f1 , f2)
	

def texture_mask( input ):
	#Needs a 3x3 window
	center_index = floor(input.size / 2)
	center = input[center_index]
	average =  ( 1 / input.size) * input.sum

	return abs(center - average)

def dilation_filter( input ):
	if input.sum > 0:
		return 1
	else:
		return 0

def edge_mask( input ):
	#not sure if we want this to be a generic filter, or just 
	#apply a couple masks to an image
	#We first apply the laplacian filter
	laplace_output = laplace(input)

	#The paper says to use the Canny algorithm, but we'll try sobel first
	sobel_output = sobel(laplace_output)
	
	#finally, we apply the dilation filter
	edge_mask = generic_filter(sobel_output , dilation_filter , (3 , 3))

	return edge_mask

