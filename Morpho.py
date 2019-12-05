import cv2
import numpy as np
img=cv2.imread("Baby.jpg",0)
kernel=np.ones((5,5),np.uint8)
# kernel has 5x5 matrix of 1
img_erode=cv2.erode(img,kernel,iterations=1)
img_dilate=cv2.dilate(img,kernel,iterations=1)
#Iterations determine how much we erode or dilate
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#The goal with opening is to remove "false positives"
#The goal with closing is to remove "false negatives" 
cv2.imshow("Erosion",img_erode)
cv2.imshow("Dilation",img_dilate)
cv2.imshow('Opening',opening)
cv2.imshow('Closing',closing)
