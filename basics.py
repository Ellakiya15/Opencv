import cv2 as cv

img = cv.imread('Photos/cat2.png')
cv.imshow('Cat',img)

#Converting to grayscale

cv.waitKey(0)