import cv2 as cv

img = cv.imread('Photos/cat1.png')
cv.imshow('Cat',img)

#Converting to grayscale
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray',gray)

#Blur
blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT) #always should be in odd
cv.imshow('Blur',blur)

#Edge Cascade
# edge = cv.Canny(img, 125,175 )
# cv.imshow('Norm canny',edge)

edge = cv.Canny(blur, 125,175 ) #to reduce edge use blur image
cv.imshow('Canny',edge)

#Dilating the Image
dilated = cv.dilate(edge, (5,5), iterations=3)
cv.imshow('Dilated',dilated)

#Eroding
eroded = cv.erode(dilated,(5,5), iterations=3)
cv.imshow('Eroded',eroded)

#Resize 
resize = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC) #TO scale the image cv.INTER_CUBIC
cv.imshow('Resize',resize)

#Cropping
crop = img[50:100, 100:120]
cv.imshow('Cropped',crop)
cv.waitKey(0)