import cv2
import numpy as np

cap =cv2.VideoCapture('video.avi')

m = cv2.createBackgroundSubtractorMOG2() #Background Subtraction

#Filtering of Frames
def filter_mask(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel)
    
    opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    
    dilation = cv2.dilate(opening,kernel,iterations = 2)

    _ ,th = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return th


#Rectangles drawn on the Vehicle detected
def draw_contours(img,frame,min_contour_w = 35, min_contour_h = 35):
    
    im2,contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if (cv2.contourArea(c)<500):
            continue

        (x,y,w,h) = cv2.boundingRect(c)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('final',frame)

while True:
    ret, frame = cap.read()

    n = m.apply(frame)

    l = filter_mask(n)
    
    draw_contours(l,frame)
    
    k = cv2.waitKey(30)
    if k==27:
        break
    
cap.release()
cv2.destroyAllWindows()


