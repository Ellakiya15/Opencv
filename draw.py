import cv2 as cv
import numpy as np

blank_img = np.zeros((500,500,3),dtype='uint8')     #np.zeros((width,height,shape))
cv.imshow('Blank',blank_img)
# img = cv.imread('Photos/cat3.png')
# cv.imshow('Normal',img)

#Painting the blank image
# blank_img[200:400, 400:500] = 0,255,0    #blank_img[200:400, 400:500] ==> [width pixcel size , height pixcel size]
# cv.imshow('Green',blank_img)

# blank_img[:] = 255,0,0
# cv.imshow('Blue',blank_img)

# blank_img[:] = 0,0,255
# cv.imshow('Red',blank_img)

# cv.rectangle(blank_img, (0,0), (250,250), (255,0,0), thickness=-1)
# #size = (250,250)==> (width,height)
# # also written as blank_img.shape[1]//2,blank_img.shape[0]//2
# # #thickness=2 (only border) thickness=-1(whole rect colored)
# cv.imshow('Rectangle_blue',blank_img)

# #Draw circle
# cv.circle(blank_img, (250,250),40 , (0,100,0), thickness=-1)
# # cv.circle(blank_img, (250,250),40 , (255,0,0), thickness=-1)
# cv.imshow('Blue_circle',blank_img)

# #Draw a line
# cv.line(blank_img, (50,150), (250,250), (255,255,255), thickness=3)
# cv.imshow('Line',blank_img)

#Text
cv.putText(blank_img, 'This is Ellakiya',(0,225), cv.FONT_HERSHEY_SIMPLEX,1.2, (255,0,0), 2)  
cv.imshow('Text',blank_img)
cv.waitKey(0)

