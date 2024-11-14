import cv2 as cv

img = cv.imread('Photos/cat2.png')
cv.imshow('Cat2',img)
def rescaleFrame(frame,scale=0.75):
    # this will work for images, videos and live videos
    width = int(frame.shape[1] * scale)  # 1 is for width of the frame
    height = int(frame.shape[0] * scale) # 0 is for height for the frame
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
# resized_image = rescaleFrame(img)
# cv.imshow('Resized',resized_image)
def changeRes(width,height):    #only works for live videos not photos
    #Live videos
    capture.set(3,width)
    capture.set(4,height)
capture = cv.VideoCapture('Videos/dog1.mp4')
while True:
        isTrue , frame = capture.read()
        frame_resized = rescaleFrame(frame)
        cv.imshow('Video',frame)
        cv.imshow('Video_resized',frame_resized)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
        
capture.release()
cv.destroyAllWindows()

cv.waitKey(0)