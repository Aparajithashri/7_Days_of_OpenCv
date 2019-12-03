# 7_DAYS OF OPENCV
   Let's get started with OpenCv. :wink:
   
# Prerequisite
  * Python - https://www.python.org/
  * Numpy - https://www.lfd.uci.edu/~gohlke/pythonlibs/ 
  * Matplotlib - https://www.lfd.uci.edu/~gohlke/pythonlibs/
  
  You can also download OpenCv in this link.

# DAY 1
## Importing
First import the packages and go through the code.
```
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. Reading and displaying the Image:

img=cv2.imread("Baby.jpg")
cv2.imshow('1st Image',img)

# 2. Resizing:

resized_img=cv2.resize(img,(500,500))
print(img.shape)
print(resized_img.shape)
#Output : (1500,1000,3)  (500,500,3)
          
# 3. Slicing image (Region of interest):

roi=img[60:160,320:420]
# Crop of the image

# 4. Rotating an image:

r,c,w=img.shape
m=cv2.getRotationMatrix2D((c/2,r/2),90,1)
rotated=cv2.warpAffine(img,m,(c,r))
# image is rotated about 90 degree

# 5. Copying an image

copied_img=img.copy()

# 6. Drawing on an image

cv2.rectangle(img,(100,300),(200,400),(0,255,0),2)
cv2.line(img,(60,20),(400,200),(0,128,128),1)
# 2nd argument - Starting point , 3rd - Ending point , 4th - Color , 5th - Thickness
cv2.circle(img,(300,150),20,(0,0,255),-1)
# 2nd argument - Center , 3rd - Radius , 4th - Color , 5th - Thickness

# 7. Writing on an image

cv2.putText(img,"New Born Baby",(50,50),cv2.FONT_ITALIC,0.5,(0,255,0),3)
# 2nd argument - String to be written , 3rd - Starting point , 3rd - Font , 4th - scale , 5th - Color , 6th - Thickness

# 8. Saving an image to the system

cv2.imwrite('C:/Users/Desktop/BABY.jpg',img)

```
# Day 2
Let's we see how to process a video :video_camera:.Download the video.py file and execute the code.
```
import numpy as np
import cv2

cap=cv2.VideoCapture(0)
# Return video from the first webcam on your computer

# To save the video
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('Output.avi',fourcc,20.0,(640,480))

while(True):
   ret,frame=cap.read()
   # Code initiates infinite loop and return each frame using read function
   
   # Save each frame on to the system
   out.write(frame)
   
   # To show the frame
   cv2.imshow('Frame',frame)
   
   # To exit from the loop
   if cv2.waitKey(1) & 0xFF==ord("q"):
      break
      
# Release the webcam and closing the window
cap.release()
out.release()
cv2.destroyAllWindows()
```
