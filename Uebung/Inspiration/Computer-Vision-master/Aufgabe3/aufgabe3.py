import numpy as np
from scipy.misc import imread, imsave, toimage
from scipy import ndimage
# import matplotlib.pyplot as plt

#~ import scipy

# Bildpunkte
# links unten 347, 432
# rechts unten 648, 423
# rechts oben 522, 330
# links oben 366, 335
def main():
    gGebaeude()


def fGebaeude():
    
    
    
    # Bild 1
    # img1 = imread('3_1.jpg') # 75 Pixel pro Fenster
    # height1 = 600
    # width1 = 600
    # pass_points1 = np.array([[437,295],[422,588],[760,604],[747,287]])
    # world_points1 = np.array([[0,0],[0,height1],[width1,height1],[width1,0]])
    # Bild 2
    img2 = imread('3_2.jpg') # 80 pro Fenster
    width2 = 720
    height2 = 600
    shift2 = 0
    pass_points2 = np.array([[474,240],[478,600],[973,588],[927,270]])#[868,552],[838,250]])
    world_points2 = np.array([[0 + shift2,0],[0 + shift2,height2],[width2 + shift2,height2],[width2 + shift2,0]])
    # Bild 3e
    img3 = imread('3_3.jpg')
    width3 = 960
    height3 = 600
    shift3 = 400
    pass_points3 = np.array([[119,184],[73,587],[906,580],[873,263]])
    world_points3 = np.array([[0 + shift3,0],[0 + shift3,height3],[width3 + shift3,height3],[width3 + shift3,0]])
    # Bild 4
    # img4 = imread('3_4.jpg')
    width4 = 525
    height4 = 600
    shift4 = 825
    pass_points4 = np.array([[90,150],[40,520],[533,533],[534,257]])
    world_points4 = np.array([[0 + shift4,0],[0 + shift4,height4],[width4 + shift4,height4],[width4 + shift4,0]])

    img = [img2, img3]
    pass_points = [pass_points2, pass_points3]
    world_points = [world_points2, world_points3]
    stitchedImage1, weightStitched1 = stitching(img, pass_points, world_points)
    imsave('stitched1.jpg', stitchedImage1)
    imsave('weight_stitched1.jpg', weightStitched1)

    # img = [stitchedImage1 , img3]
    # pass_points = [pass_points1, pass_points3]
    # world_points = [world_points1, world_points3]
    # stitchedImage2, weight2 = stitching(img, pass_points, world_points,  stitchedWeight=weightStitched1)
    # imsave('stitched2.jpg', stitchedImage2)
    # imsave('weight_stitched2.jpg', weight2)

    # img = [stitchedImage2 , img4]
    # pass_points = [pass_points1, pass_points4]
    # world_points = [world_points1, world_points4]
    # stitchedImage3, weight3 = stitching(img, pass_points, world_points,  stitchedWeight=weightStitched2)
    # imsave('stitched2.jpg', stitchedImage3)
    # imsave('weight_stitched2.jpg', weight3)





def gGebaeude():
    img1 = imread('2_1.jpg')
    img2 = imread('2_2.jpg')
    img3 = imread('2_3.jpg')
    # img2 = imread('2.jpg')

    # width = 800.0
    # height = 700.0
    # world_points = np.array([[0.0,0.0],[0.0,height],[width,height],[width, 0.0],[250,0], [250, height]]) # Bild1
    # world_points = np.array([[0.0,0.0],[0.0,height],[200.0,0.0], [200.0, height],[width,height],[width, 0.0]]) # Bild1
    # pass_points = np.array([[446,74],[445,689],[770,693],[781,74],[298,74], [305,260],[305,330],[304,690]]) # Bild 1 
    # pass_points = np.array([[0,69],[0,688],[773,693],[781,74],[298,72], [304,689]]) # Bild 1 
    # pass_points = np.array([[45,69],[33,692],[298,72],[304,689],[773,693],[781,74]]) # Bild 1 
    # world_points2 = np.array([[0.0,0.0],[0.0,height],[width,height],[width, 0.0],[width/2,height/2]]) # Bild2
    # pass_points2 = np.array([[210,164],[210.0,275.0],[347.0,275.0],[347.0,140.0],[245,199], [247,234]]) # Bild 2

    # 75 Pixel pro Fenster
    width3 = 675
    height3 = 350
    pass_points3 = np.array([[62,234],[22,528],[620,533],[616,200]])
    world_points3 = np.array([[0,0],[0,height3],[width3,height3],[width3,0]])
    height2 = 350
    width2 = 600
    shift2 = 520
    pass_points2 = np.array([[118,182],[72,542],[665,549],[652,233]])#[868,552],[838,250]])
    world_points2 = np.array([[0 + shift2,0],[0 + shift2,height2],[width2 + shift2,height2],[width2 + shift2,0]])
    width1 = 525
    height1 = 350
    shift3 = 830
    pass_points1 = np.array([[90,150],[40,520],[533,533],[534,257]])
    world_points1 = np.array([[0 + shift3,0],[0 + shift3,height1],[width1 + shift3,height1],[width1 + shift3,0]])

    img = [img3, img2]
    pass_points = [pass_points3, pass_points2]
    world_points = [world_points3, world_points2]
    stitchedImage, weightStitched1 = stitching(img, pass_points, world_points)

    imsave('stitched1.jpg', stitchedImage)
    imsave('weight_stitched1.jpg', weightStitched1)

    img = [stitchedImage , img1]
    pass_points = [pass_points3, pass_points1]
    world_points = [world_points3, world_points1]
    stitchedImage, weight = stitching(img, pass_points, world_points,  stitchedWeight=weightStitched1)

    imsave('stitched2.jpg', stitchedImage)
    imsave('weight_stitched2.jpg', weight)




def stitching(img, pass_points, world_points, stitchedWeight=None, mode='multi_band'):

    numberImages = len(img)

    width1 = max(world_points[0][:,0])
    height1 = max(world_points[0][:,1])

    width2 = max(world_points[1][:,0])
    height2 = max(world_points[0][:,1])


    weight1 = stitchedWeight
    if stitchedWeight == None:
        weight1 = calculateWeight(img[0])
    
    weight2 = calculateWeight(img[1])

    if stitchedWeight == None:
        weight1 = projectiveRectification(weight1, world_points[0], pass_points[0], max(world_points[0][:,0]), height1)
        imsave('weight1_first.jpg', weight1)

    weight2 = projectiveRectification(weight2, world_points[1], pass_points[1], max(world_points[1][:,0]), height2)

    if stitchedWeight == None:
        imsave('weight2_first.jpg', weight2)
    else:
        imsave('weight2_second.jpg', weight2)


    img1 = img[0]
    if stitchedWeight == None:
        img1 = projectiveRectification(img[0], world_points[0], pass_points[0], max(world_points[0][:,0]), height1)
        imsave('img1_first.jpg', img1)

    img2 = projectiveRectification(img[1], world_points[1], pass_points[1], max(world_points[1][:,0]), height2)
    if stitchedWeight == None:
        imsave('img2_first.jpg', img2)
    else:
        imsave('img2_second.jpg', img2)


    #toimage(img1).show()
    #toimage(img2).show()
    #toimage(weight1).show()
    #toimage(weight2).show()

    height1 = img1.shape[0]
    width1 = img1.shape[1]
    height2 = img2.shape[0]
    width2 = img2.shape[1]

    newHeight = max(height1,height2)
    newWidth = max(max(world_points[0][:,0]),max(world_points[1][:,0]))

    stitchedImage = np.empty((newHeight, newWidth, img[0].shape[2]))
    newWeight = np.empty((newHeight, newWidth, 3))
    
    if mode == 'multi_band':
        tpass_filter = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]])
        lowPassImage1 = ndimage.uniform_filter(img1, 5, mode='reflect')
        lowPassImage2 = ndimage.uniform_filter(img2, 5, mode='reflect')
        
        highPassImage1 = np.subtract(img1, lowPassImage1)
        highPassImage2 = np.subtract(img2, lowPassImage2)

        if stitchedWeight == None:
            imsave('high_pass.jpg', highPassImage2)
            imsave('low_pass.jpg', lowPassImage2)
        else:
            imsave('high_pass2.jpg', highPassImage2)
            imsave('low_pass2.jpg', lowPassImage2)

        

    for y in xrange(newHeight):
        for x in xrange(newWidth):


            if x >= width1 or y >= height1:
                pointColor1 = np.array([0,0,0])
                pointWeight1 = 0
                pointLowPass1 = np.array([0,0,0])
                pointHighPass1 = np.array([0,0,0])
            else:
                pointHighPass1 = highPassImage1[y,x,:]
                pointLowPass1 = lowPassImage1[y,x,:]
                pointWeight1 = weight1[y,x,0]
                pointColor1 = img1[y,x,:]

            if x >= width2 or y >= height2:
                pointColor2 = np.array([0,0,0])
                pointWeight2 = 0
                pointLowPass2 = np.array([0,0,0])
                pointHighPass2 = np.array([0,0,0])
            else:
                pointHighPass2 = highPassImage2[y,x,:]
                pointLowPass2 = lowPassImage2[y,x,:]
                pointWeight2 = weight2[y,x,0]
                pointColor2 = img2[y,x,:]

            if mode == 'none':

                if pointWeight1 > pointWeight2:
                    stitchedImage[y,x,:] = img1[y,x,:]
                    newWeight[y,x] = pointWeight1
                else:
                    stitchedImage[y,x,:] = img2[y,x,:]
                    newWeight[y,x] = pointWeight2

            elif mode == 'sum':

                stitchedImage[y,x:] = (pointWeight1 * pointColor1 + pointWeight2 * pointColor2) / (pointWeight1 + pointWeight2)

                newWeight[y,x] = (pointWeight1 + pointWeight2)/2

            elif mode == 'multi_band':

                value = (pointWeight1 * pointLowPass1 + pointWeight2 * pointLowPass2) / (pointWeight1 + pointWeight2)
                if pointWeight1 > pointWeight2:
                    value = value + pointHighPass1
                else:
                    value = value + pointHighPass2
                    
                stitchedImage[y,x,:] = value


    return stitchedImage, newWeight




def calculateWeight(img):

    dimY,dimX,_ = img.shape
    center = [dimY/2,dimX/2]
    weight = np.empty((dimY,dimX,3), dtype=np.float64)

    yWeight1 = np.array([])
    yWeight2 = np.array([])
    xWeight1 = np.array([])
    xWeight2 = np.array([])
    increase_step = 255/(dimY/2.0)

    for i in np.arange(0, dimY/2):
        if i == dimY/2 -1:
            yWeight1 = np.append(yWeight1, np.ceil(increase_step*i))
        else:
            yWeight1 = np.append(yWeight1, np.round(increase_step*i))

    for i in np.arange(dimY/2, 0, -1):
        if i == 1:
            yWeight2 = np.append(yWeight2, np.floor(increase_step*i))
        else:
            yWeight2 = np.append(yWeight2, np.round(increase_step*i))

    yWeight = np.concatenate((yWeight1, yWeight2), axis=0)

    for i in np.arange(0, dimX/2):
        if i == dimX/2 -1:
            xWeight1 = np.append(xWeight1, np.ceil(increase_step*i))
        else:
            xWeight1 = np.append(xWeight1, np.round(increase_step*i))

    for i in np.arange(dimX/2, 0, -1):
        if i == 1:
            xWeight2 = np.append(xWeight2, np.floor(increase_step*i))
        else:
            xWeight2 = np.append(xWeight2, np.round(increase_step*i))

    xWeight = np.concatenate((xWeight1, xWeight2), axis=0)

    weight = np.outer(yWeight, xWeight)

    weight3d = np.empty((weight.shape[0],weight.shape[1],3))

    weight3d[:,:,0] = weight 
    weight3d[:,:,1] = weight 
    weight3d[:,:,2] = weight 

    return weight3d


def projectiveRectification(img, world_points, pass_points, width, height, interpolate='none'):

    img = img.astype(np.float64)
    img_max = img.max()
    img = img / img_max

    img_transform = np.zeros((height, width, img.shape[2]))#, dtype=np.float64)

    matrix = np.empty((0,8))
    real_coord = np.array([])

    for i in xrange(len(world_points)):
        x, y = world_points[i]
        x_, y_ = pass_points[i]


        matrix = np.vstack((matrix, [x_, y_, 1, 0, 0, 0, -x*x_, -x*y_]))
        matrix = np.vstack((matrix, [0,0,0, x_, y_, 1, -y*x_, -y*y_]))
        real_coord = np.append(real_coord, [x,y])

    
    real_coord = np.transpose(real_coord)
    matrix_inv = np.linalg.pinv(matrix)
    vector = matrix_inv.dot(real_coord)
    # exit()
    a1 = vector[0]
    a2 = vector[1]
    a3 = vector[2]
    b1 = vector[3]
    b2 = vector[4]
    b3 = vector[5]
    c1 = vector[6]
    c2 = vector[7]

    dimY,dimX,_ = img_transform.shape
    center = [dimY/2,dimX/2]
    weight = np.empty((dimY,dimX), dtype=np.float64)

    for y in np.arange(0, img_transform.shape[0]):
        for x in np.arange(0, img_transform.shape[1]):
            
            denominator = (b1*c2 - b2*c1)*x + (a2*c1-a1*c2)*y + a1*b2 - a2*b1
            index_x = ((b2-c2*b3)*x + (a3*c2 - a2)*y + a2*b3 - a3*b2) / denominator
            index_y = ((b3*c1 - b1)*x + (a1 - a3*c1)*y + a3*b1 - a1*b3) / denominator

            if index_x > 0 and index_x < img.shape[1] and index_y > 0 and index_y < img.shape[0]:
                
                p1 = np.array([np.sqrt((np.floor(index_x) - index_x)**2 + (np.floor(index_y) - index_y)**2), np.floor(index_y), np.floor(index_x)]) 
                p2 = np.array([np.sqrt((np.ceil(index_x) - index_x)**2 + (np.floor(index_y) - index_y)**2), np.floor(index_y), np.ceil(index_x)]) 
                p3 = np.array([np.sqrt((np.floor(index_x) - index_x)**2 + (np.ceil(index_y) - index_y)**2), np.ceil(index_y), np.floor(index_x)]) 
                p4 = np.array([np.sqrt((np.ceil(index_x) - index_x)**2 + (np.ceil(index_y) - index_y)**2), np.ceil(index_y), np.ceil(index_x)]) 



                if interpolate == 'bilinear':
                    area1 = (index_x-np.floor(index_x)) * (index_y-np.floor(index_y))
                    area2 = (index_x-np.ceil(index_x)) * (np.floor(index_y)-index_y)
                    area3 = (np.floor(index_x)-index_x) * (index_y-np.ceil(index_y))
                    area4 = (np.ceil(index_x)-index_x) * (np.ceil(index_y)-index_y)
                    
                    p1[1], p1[2] = cut(p1[1], p1[2], img.shape)
                    p2[1], p2[2] = cut(p2[1], p2[2], img.shape)
                    p3[1], p3[2] = cut(p3[1], p3[2], img.shape)
                    p4[1], p4[2] = cut(p4[1], p4[2], img.shape)

                    value = area4*img[ p1[1],p1[2] ] + area3*img[ p2[1],p2[2] ] + area2*img[ p3[1],p3[2] ] + area1*img[ p4[1],p4[2] ]
                    
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
                    

    # toimage(img_transform).show()

    return img_transform#, weight





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
    # img_transform = projectiveRectification(interpolate='none')
    # imsave('luftbild_none.jpg', img_transform)
    # img_transform = projectiveRectification(interpolate='nearest')
    # imsave('luftbild_nearest.jpg', img_transform)
    # img_transform = projectiveRectification(interpolate='bilinear')
    # imsave('luftbild_bilinear.jpg', img_transform)
