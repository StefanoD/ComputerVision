import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.misc



def main():
	test5()



def test1():
	img = scipy.misc.imread('images/gletscher.jpg')
	angle = np.radians(0)
	A = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
	print A
	alpha = (0,1)
	
	# interpolate = bilinear, nearest
	img_none = affine_transformation(A,alpha,img, interpolate='none')
	scipy.misc.imsave('images/test1_gletscher_none.jpg', img_none)
	img_nearest = affine_transformation(A,alpha,img, interpolate='nearest')
	scipy.misc.imsave('images/test1_gletscher_nearest.jpg', img_nearest)
	img_bilinear = affine_transformation(A,alpha,img, interpolate='bilinear')
	scipy.misc.imsave('images/test1_gletscher_bilinear.jpg', img_bilinear)

def test2():
	img = scipy.misc.imread('images/gletscher.jpg')
	angle = np.radians(30)
	A = 0.7 * np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
	print A
	alpha = (0,0)
	
	# interpolate = bilinear, nearest
	img_none = affine_transformation(A,alpha,img, interpolate='none')
	scipy.misc.imsave('images/test2_gletscher_none.jpg', img_none)
	img_nearest = affine_transformation(A,alpha,img, interpolate='nearest')
	scipy.misc.imsave('images/test2_gletscher_nearest.jpg', img_nearest)
	img_bilinear = affine_transformation(A,alpha,img, interpolate='bilinear')
	scipy.misc.imsave('images/test2_gletscher_bilinear.jpg', img_bilinear)

def test3():
	img = scipy.misc.imread('images/gletscher.jpg')
	angle = np.radians(30)
	stretch_x = 0.8
	stretch_y = 1.2
	scale = 0.7
	A = scale * np.array([[ stretch_x * np.cos(angle), stretch_y * np.sin(angle)],[stretch_x * -np.sin(angle), stretch_y * np.cos(angle)]])
	print A
	alpha = (0,0)
	
	# interpolate = bilinear, nearest
	img_none = affine_transformation(A,alpha,img, interpolate='none')
	scipy.misc.imsave('images/test3_gletscher_none.jpg', img_none)
	img_nearest = affine_transformation(A,alpha,img, interpolate='nearest')
	scipy.misc.imsave('images/test3_gletscher_nearest.jpg', img_nearest)
	img_bilinear = affine_transformation(A,alpha,img, interpolate='bilinear')
	scipy.misc.imsave('images/test3_gletscher_bilinear.jpg', img_bilinear)

def test4():
	img = scipy.misc.imread('images/gletscher.jpg')
	angle = np.radians(30)
	stretch_x = 0.8
	stretch_y = 1.2
	scale = 0.7
	scale_diagonal = 1.5
	scale_orthogonal = 0.5
	A = scale * np.array([[ stretch_x * np.cos(angle), scale_diagonal * stretch_y * np.sin(angle)],[scale_orthogonal * stretch_x * -np.sin(angle), stretch_y * np.cos(angle)]])
	print A
	alpha = (0,0)
	
	# interpolate = bilinear, nearest
	img_none = affine_transformation(A,alpha,img, interpolate='none')
	scipy.misc.imsave('images/test4_gletscher_none.jpg', img_none)
	img_nearest = affine_transformation(A,alpha,img, interpolate='nearest')
	scipy.misc.imsave('images/test4_gletscher_nearest.jpg', img_nearest)
	img_bilinear = affine_transformation(A,alpha,img, interpolate='bilinear')
	scipy.misc.imsave('images/test4_gletscher_bilinear.jpg', img_bilinear)

def test5():
	img = scipy.misc.imread('images/ambassadors.jpg')
	angle = np.radians(-27)
	stretch_x = 9.0
	stretch_y = 1.5
	scale = 1.0
	scale_diagonal = 0.4
	scale_orthogonal = 0.9
	A = scale * np.array([[ stretch_x * np.cos(angle), scale_diagonal * stretch_y * np.sin(angle)],[scale_orthogonal * stretch_x * -np.sin(angle), stretch_y * np.cos(angle)]])
	print A
	alpha = (0,-200)
	
	# interpolate = bilinear, nearest
	img_none = affine_transformation(A,alpha,img, interpolate='none')
	scipy.misc.imsave('images/test5_ambassadors_none.jpg', img_none)
	img_nearest = affine_transformation(A,alpha,img, interpolate='nearest')
	scipy.misc.imsave('images/test5_ambassadors_nearest.jpg', img_nearest)
	img_bilinear = affine_transformation(A,alpha,img, interpolate='bilinear')
	scipy.misc.imsave('images/test5_ambassadors_bilinear.jpg', img_bilinear)


def affine_transformation(A, alpha, img, interpolate="nearest"):
	img = img.astype(np.float64)
	img_max = img.max()
	img = img / img_max

	img_transform = np.zeros((img.shape[0],img.shape[1], img.shape[2]))#, dtype=np.float64)

	A_inv = np.linalg.inv(A)

	for y in np.arange(0, img_transform.shape[0]):
		for x in np.arange(0, img_transform.shape[1]):
			transform_point = [y - img_transform.shape[0]/2,x - img_transform.shape[1]/2]
			point = np.dot(transform_point,A_inv)
			#print point




			index_x = point[1]  + img_transform.shape[1]/2 - alpha[0] #512
			index_y = point[0]  + img_transform.shape[0]/2 - alpha[1] #512

			#if index_y < img.shape[0] and index_x < img.shape[1]:
			if index_x > 0 and index_x < img.shape[1] and index_y > 0 and index_y < img.shape[0]:

				p1 = np.array([np.sqrt((np.floor(index_x) - index_x)**2 + (np.floor(index_y) - index_y)**2), np.floor(index_y), np.floor(index_x)]) 
				p2 = np.array([np.sqrt((np.ceil(index_x) - index_x)**2 + (np.floor(index_y) - index_y)**2), np.floor(index_y), np.ceil(index_x)]) 
				p3 = np.array([np.sqrt((np.floor(index_x) - index_x)**2 + (np.ceil(index_y) - index_y)**2), np.ceil(index_y), np.floor(index_x)]) 
				p4 = np.array([np.sqrt((np.ceil(index_x) - index_x)**2 + (np.ceil(index_y) - index_y)**2), np.ceil(index_y), np.ceil(index_x)]) 



				if interpolate == 'bilinear':
					a1 = (index_x-np.floor(index_x)) * (index_y-np.floor(index_y))
					a2 = (index_x-np.ceil(index_x)) * (np.floor(index_y)-index_y)
					a3 = (np.floor(index_x)-index_x) * (index_y-np.ceil(index_y))
					a4 = (np.ceil(index_x)-index_x) * (np.ceil(index_y)-index_y)
					# print index_x,index_y, np.ceil(index_x), np.floor(index_x)

					# print x,y
					

					# print a1
					# print a2
					# print a3
					# print a4

					#print a1, a2, a3, a4
					
					p1[1], p1[2] = cut(p1[1], p1[2], img.shape)
					p2[1], p2[2] = cut(p2[1], p2[2], img.shape)
					p3[1], p3[2] = cut(p3[1], p3[2], img.shape)
					p4[1], p4[2] = cut(p4[1], p4[2], img.shape)



					value = a4*img[ p1[1],p1[2] ] + a3*img[ p2[1],p2[2] ] + a2*img[ p3[1],p3[2] ] + a1*img[ p4[1],p4[2] ]
					# print value
					# print value
					if a1 == 0 and a2 == 0 and a3 == 0 and a4 == 0:
						img_transform[y  ,x  ,:] = img[index_y,index_x,:]
					else:	
						img_transform[y  ,x  ,:] = value
					# img_transform[y  ,x  ,:] = value

				elif interpolate == 'nearest':
					points = np.array([p1,p2,p3,p4])
					distance = points[:,0]
					index_nearest = np.where(distance==distance.min())
					nearest_x = points[index_nearest[0][0]][2]
					nearest_y = points[index_nearest[0][0]][1]

					nearest_y, nearest_x = cut(nearest_y, nearest_x, img.shape)

					img_transform[y  ,x  ,:] = img[nearest_y,nearest_x,:]

				elif interpolate == 'none':
					img_transform[y  ,x  ,:] = img[index_y,index_x,:]

	return img_transform

def cut(y,x,shape):

	if y <= 0:
		y = 0
	if y >= shape[0]:
		y = shape[0] - 1

	if x <= 0:
		x = 0
	if x >= shape[1]:
		x = shape[1] - 1

	return y, x

if __name__ == '__main__':
	main()