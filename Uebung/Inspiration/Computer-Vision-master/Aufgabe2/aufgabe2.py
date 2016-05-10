import numpy as np
from scipy.misc import imread, imsave, toimage
#~ import scipy

# Bildpunkte
# links unten 347, 432
# rechts unten 648, 423
# rechts oben 522, 330
# links oben 366, 335

def projectiveRectification(interpolate='none'):

    img = imread('schraegbild_tempelhof.jpg')

    # toimage(img).show()
    #~ interpolate = 'none'
    #~ interpolate = 'nearest'
    #~ interpolate = 'bilinear'
    img = img.astype(np.float64)
    img_max = img.max()
    img = img / img_max

    width = 600.0
    height = 900.0
    img_transform = np.zeros((height, width, img.shape[2]))#, dtype=np.float64)



    # A_inv = np.linalg.inv(A)

#~ flugfeld
    # x1 = 230
    # y1 = 332
    # x2 = 190
    # y2 = 460
    # x3 = 600
    # y3 = 430
    # x4 = 540
    # y4 = 320

    # x1 = 170
    # y1 = 293
    # x2 = 170
    # y2 = 713
    # x3 = 971
    # y3 = 692
    # x4 = 515
    # y4 = 288

    x1 = 344.0
    y1 = 334.0
    x2 = 300.0
    y2 = 456.0
    x3 = 690.0
    y3 = 430.0
    x4 = 548.0
    y4 = 330.0



    matrix = np.array([[x1, y1, 1, 0, 0, 0, -0.0*x1, -0.0*y1],
            [0,0,0, x1, y1, 1, -0.0*x1, -0.0*y1],
            [x2, y2, 1, 0, 0, 0, -0.0*x2, -0.0*y2],
            [0,0,0, x2, y2, 1, -height*x2, -height*y2],
            [x3, y3, 1, 0, 0, 0, -width*x3, -width*y3],
            [0,0,0, x3, y3, 1, -height*x3, -height*y3],
            [x4, y4, 1, 0, 0, 0, -width*x4, -width*y4],
            [0,0,0, x4, y4, 1, -0.0*x4, -0.0*y4]])

    

    real_coord = np.transpose(np.array([0.0, 0.0, 0.0, height, width, height, width, 0.0]))
    matrix_inv = np.linalg.inv(matrix)
    vector = matrix_inv.dot(real_coord)
    a1 = vector[0]
    a2 = vector[1]
    a3 = vector[2]
    b1 = vector[3]
    b2 = vector[4]
    b3 = vector[5]
    c1 = vector[6]
    c2 = vector[7]

    #~ exit()

    for y in np.arange(0, img_transform.shape[0]):
        for x in np.arange(0, img_transform.shape[1]):
            
            # index_x = 0
            # index_y = 0
            
            denominator = (b1*c2 - b2*c1)*x + (a2*c1-a1*c2)*y + a1*b2 - a2*b1
            index_x = ((b2-c2*b3)*x + (a3*c2 - a2)*y + a2*b3 - a3*b2) / denominator
            index_y = ((b3*c1 - b1)*x + (a1 - a3*c1)*y + a3*b1 - a1*b3) / denominator

    

            #~ print 'point coords', y,x
            #~ print 'index_y, index_x',index_y, index_x
            #~ exit()
            #if index_y < img.shape[0] and index_x < img.shape[1]:
            #~ print 'index_y, index_x',index_y, index_x
            #~ exit()
            if index_x > 0 and index_x < img.shape[1] and index_y > 0 and index_y < img.shape[0]:
                #~ print 'index_y, index_x in der if',index_y, index_x
                #~ print 'point coords', y,x
                
                p1 = np.array([np.sqrt((np.floor(index_x) - index_x)**2 + (np.floor(index_y) - index_y)**2), np.floor(index_y), np.floor(index_x)]) 
                p2 = np.array([np.sqrt((np.ceil(index_x) - index_x)**2 + (np.floor(index_y) - index_y)**2), np.floor(index_y), np.ceil(index_x)]) 
                p3 = np.array([np.sqrt((np.floor(index_x) - index_x)**2 + (np.ceil(index_y) - index_y)**2), np.ceil(index_y), np.floor(index_x)]) 
                p4 = np.array([np.sqrt((np.ceil(index_x) - index_x)**2 + (np.ceil(index_y) - index_y)**2), np.ceil(index_y), np.ceil(index_x)]) 



                if interpolate == 'bilinear':
                    area1 = (index_x-np.floor(index_x)) * (index_y-np.floor(index_y))
                    area2 = (index_x-np.ceil(index_x)) * (np.floor(index_y)-index_y)
                    area3 = (np.floor(index_x)-index_x) * (index_y-np.ceil(index_y))
                    area4 = (np.ceil(index_x)-index_x) * (np.ceil(index_y)-index_y)
                    #~ print index_x,index_y
                    # print x,y
                    

                    #~ print a1
                    #~ print a2
                    #~ print a3
                    #~ print a4

                    
                    
                    p1[1], p1[2] = cut(p1[1], p1[2], img.shape)
                    p2[1], p2[2] = cut(p2[1], p2[2], img.shape)
                    p3[1], p3[2] = cut(p3[1], p3[2], img.shape)
                    p4[1], p4[2] = cut(p4[1], p4[2], img.shape)

                    value = area4*img[ p1[1],p1[2] ] + area3*img[ p2[1],p2[2] ] + area2*img[ p3[1],p3[2] ] + area1*img[ p4[1],p4[2] ]
                    
                    #~ # print value
                    if np.sum(value) == 0:
                        img_transform[y  ,x  ,:] = img[index_y,index_x,:]
                    else:    
                        img_transform[y  ,x  ,:] = value

                elif interpolate == 'nearest':
                    points = np.array([p1,p2,p3,p4])
                    distance = points[:,0]
                    index_nearest = np.where(distance==distance.min())
                    nearest_x = points[index_nearest[0][0]][2]
                    nearest_y = points[index_nearest[0][0]][1]

                    nearest_y, nearest_x = cut(nearest_y, nearest_x, img.shape)

                    #~ print img[nearest_y,nearest_x,:]
                    img_transform[y  ,x  ,:] = img[nearest_y,nearest_x,:]

                elif interpolate == 'none':
                    img_transform[y  ,x  ,:] = img[index_y,index_x,:]
                    

    #~ toimage(img_transform).show()

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
    
    img_transform = projectiveRectification(interpolate='none')
    imsave('luftbild_none.jpg', img_transform)
    img_transform = projectiveRectification(interpolate='nearest')
    imsave('luftbild_nearest.jpg', img_transform)
    img_transform = projectiveRectification(interpolate='bilinear')
    imsave('luftbild_bilinear.jpg', img_transform)
