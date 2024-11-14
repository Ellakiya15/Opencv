import cv2 as cv
import numpy as np

img = cv.imread('Photos/image.png')
cv.imshow('Boston',img)

def translate(img, x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> Left
#-y --> Up
# x --> Right
# y --> Down

# translated = translate(img,-100,100) #-x,+y
# cv.imshow('Translated', translated)

#Rotation
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]
    
    if rotPoint is None:
        rotPoint = (width//2, height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)
    
    return cv.warpAffine(img,rotMat,dimensions)

# rotated = rotate(img, 90)
# # cv.imshow('Rotated', rotated)

# flip = cv.flip(img,1) #0 -> vertical flip, 1-> horizontal flip, -1 --> vertical as well as horizontal
# cv.imshow('Flip',flip)

# Cropped
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)
cv.waitKey(0)