'''
This code enables you to detect edges in an image using a method called "FuzzEdge".
@author: Abder-Rahman Ali
abder@cs.stir.ac.uk
'''

import numpy
import cv2
import os
import math
import argparse as ap

start_to_DKend = []
start_to_DKend_pixel_frequency = []
BRbegin_to_end = []
BRbegin_to_end_pixel_frequency = []
BRbegin_to_end_reversed_list = []
gray_level_frequency_DKbegin_DKend = []
gray_levels_DKbegin_DKend = []
gray_levels_MDbegin_MDend = []
gray_level_frequency_MDbegin_MDend = []
gray_levels_BRbegin_BRend = []
gray_level_frequency_BRbegin_BRend = []
threshold = 0

parser = ap.ArgumentParser()
parser.add_argument('-ip', '--inputPath', help='Where the images you would like to detect the edges for reside', required='True')
parser.add_argument("-op", "--outputPath", help='Where to save the results', required='True')
args = vars(parser.parse_args())

input_path = args['inputPath']
results = args['outputPath']

for root, dirs, files in os.walk(input_path):
	for file in files:
		img = cv2.imread(root + '/' + file) 
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# get the number of gray levels L
		gray_levels = numpy.unique(img)
		L = len(gray_levels)
		# number of fuzzy sets (concepts) = 3 (dark, median, bright)
		Nf = 3
		left_overlap = 0
		right_overlap = 0

		DK_end = (L-1)/Nf
		BR_begin = (Nf-1) * DK_end
		MD_begin = DK_end - left_overlap
		MD_end = BR_begin + right_overlap

		for i in range(0,int(DK_end)):
			if (img == i).sum() >= threshold:
				DK_begin = i
				break

		DK_begin = 0

		for i in range(int(BR_begin),L-1):
			BRbegin_to_end.append(i)
			BRbegin_to_end_pixel_frequency.append((img == i).sum())

		BRbegin_to_end_reversed = reversed(BRbegin_to_end)

		for i in BRbegin_to_end_reversed:
			BRbegin_to_end_reversed_list.append(i)	

		idx = 0
		for i in reversed(BRbegin_to_end_pixel_frequency):
			if i >= threshold:
				BR_end = BRbegin_to_end_reversed_list[idx]
				break
			else:
				idx = idx + 1

		# working on the interval [DK_begin,DK_end]
		for i in range(int(DK_begin),int(DK_end+1)):
			gray_levels_DKbegin_DKend.append(i)

		for i in range(int(DK_begin),int(DK_end+1)):
			gray_level_frequency_DKbegin_DKend.append((img == i).sum())

		# number of pixels
		a,b = numpy.shape(img)
		number_of_pixels = a*b

		pixel_histogram = [float(x)/float(number_of_pixels) for x in gray_level_frequency_DKbegin_DKend]
		max_index, max_value = max(enumerate(pixel_histogram), key=lambda item: item[1])
		gray_level_with_maximum_histogram = gray_levels_DKbegin_DKend[max_index]

		# generate the membership function of fDK of the fuzzy set DK
		mDK = gray_level_with_maximum_histogram
		alphaDK = mDK - DK_begin
		betaDK = DK_end - mDK

		# working on the interval [MD_begin,MD_end]
		for i in range(int(MD_begin),int(MD_end+1)):
			gray_levels_MDbegin_MDend.append(i)
	
		for i in range(int(MD_begin),int(MD_end+1)):
			gray_level_frequency_MDbegin_MDend.append((img == i).sum())

		pixel_histogram = [float(x)/float(number_of_pixels) for x in gray_level_frequency_MDbegin_MDend]
		max_index, max_value = max(enumerate(pixel_histogram), key=lambda item: item[1])
		gray_level_with_maximum_histogram = gray_levels_MDbegin_MDend[max_index]

		# generate the membership function of fMD of the fuzzy set MD
		mMD = gray_level_with_maximum_histogram
		alphaMD = mMD - MD_begin
		betaMD = MD_end - mMD

		# working on the interval [BR_begin,BR_end]
		for i in range(int(BR_begin),int(BR_end+1)):
			gray_levels_BRbegin_BRend.append(i)

		for i in range(int(BR_begin),int(BR_end+1)):
			gray_level_frequency_BRbegin_BRend.append((img == i).sum())

		pixel_histogram = [float(x)/float(number_of_pixels) for x in gray_level_frequency_BRbegin_BRend]
		max_index, max_value = max(enumerate(pixel_histogram), key=lambda item: item[1])
	
		gray_level_with_maximum_histogram = gray_levels_BRbegin_BRend[max_index]

		# generate the membership function of fBR of the fuzzy set BR
		mBR = gray_level_with_maximum_histogram
		alphaBR = mBR - BR_begin
		betaBR = BR_end - mBR

		# create the LR-number
		def LR(x,m,alpha,beta):
			if x <= m:
				y = (m-x)/alpha
				L = max(0,1-y)
		
				return L
			elif x >= m:
				y = (x-m)/beta
				R = max(0,1-y)
				return R

		# create the fuzzy interval
		m_l = DK_begin
		m_r = BR_end

		def LR_I(x,ml,mr):
			if x <= ml:
				#L = 0
				L = 0
				return L
			elif x >= ml and x <= mr:
				return 1
			elif x >= mr:
				#R = 0
				R = 0
				return R

		# set the kernels (sample window) for the 3 fuzzy sets, and for the
		# fuzzy estimator (FE)
		def kernel_DK(x):
			result = numpy.array([[LR(x,mDK,alphaDK,betaDK),LR(x,mDK,alphaDK,betaDK),LR(x,mDK,alphaDK,betaDK)],[LR(x,mDK,alphaDK,betaDK),LR(x,mDK,alphaDK,betaDK),LR(x,mDK,alphaDK,betaDK)],[LR(x,mDK,alphaDK,betaDK),LR(x,mDK,alphaDK,betaDK),LR(x,mDK,alphaDK,betaDK)]])
			return result

		def kernel_MD(x):
			result = numpy.array([[LR(x,mMD,alphaMD,betaMD),LR(x,mMD,alphaMD,betaMD),LR(x,mMD,alphaMD,betaMD)],[LR(x,mMD,alphaMD,betaMD),LR(x,mMD,alphaMD,betaMD),LR(x,mMD,alphaMD,betaMD)],[LR(x,mMD,alphaMD,betaMD),LR(x,mMD,alphaMD,betaMD),LR(x,mMD,alphaMD,betaMD)]])
			return result

		def kernel_BR(x):
			result = numpy.array([[LR(x,mBR,alphaBR,betaBR),LR(x,mBR,alphaBR,betaBR),LR(x,mBR,alphaBR,betaBR)],[LR(x,mBR,alphaBR,betaBR),LR(x,mBR,alphaBR,betaBR),LR(x,mBR,alphaBR,betaBR)],[LR(x,mBR,alphaBR,betaBR),LR(x,mBR,alphaBR,betaBR),LR(x,mBR,alphaBR,betaBR)]])
			return result

		def kernel_FE(x):
			result = numpy.array([[LR_I(x,m_l,m_r),LR_I(x,m_l,m_r),LR_I(x,m_l,m_r)],[LR_I(x,m_l,m_r),LR_I(x,m_l,m_r),LR_I(x,m_l,m_r)],[LR_I(x,m_l,m_r),LR_I(x,m_l,m_r),LR_I(x,m_l,m_r)]])
			return result

		# take the image and the kernel and return their convolution
		# image is a numpy array of size [image_height,image_width]
		# kernel is a numpy array of size [kernel_height,kernel_width]
		# the output is a numpy array (convolution output/convolved image) of size [image_height,image_width]
		def convolve(image):
			y_DK = numpy.zeros_like(image)
			y_MD = numpy.zeros_like(image)
			y_BR = numpy.zeros_like(image)
			y_FE = numpy.zeros_like(image)
			# convolution output
			output = numpy.zeros_like(image)
			# add zero-padding to the input image
			image_padded = numpy.zeros((image.shape[0] + 2, image.shape[1] + 2)) 
			image_padded[1:-1, 1:-1] = image
			# loop over each pixel in the image
			for x in range(image.shape[1]):     
				for y in range(image.shape[0]):
					pixel_intensity = image[y,x]
					y_DK[y,x]= numpy.std(kernel_DK(pixel_intensity)*image_padded[y:y+3,x:x+3])
					y_MD[y,x]= numpy.std(kernel_MD(pixel_intensity)*image_padded[y:y+3,x:x+3])
					y_BR[y,x]= numpy.std(kernel_BR(pixel_intensity)*image_padded[y:y+3,x:x+3])
					y_FE[y,x]= numpy.std(kernel_FE(pixel_intensity)*image_padded[y:y+3,x:x+3])
					if abs(y_DK[y,x]-y_FE[y,x]) < abs(y_MD[y,x]-y_FE[y,x]):
						output[y,x] = 0
					else:
						output[y,x]=y_MD[y,x]*5 # multiplied by 10 to emphasize the detected edges
					if abs(y_BR[y,x]-y_FE[y,x]) < abs(output[y,x]-y_FE[y,x]):
						output[y,x] = y_BR[y,x]*5

			return output

		# convolve each kernel and the image
		convolve_img = convolve(img)
		cv2.imwrite(os.path.join(results, file + '-result.jpg'), convolve_img)
