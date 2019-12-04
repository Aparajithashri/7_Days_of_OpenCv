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
# Day 3
Let's we explore more about opencv.
## 1.Color Filtering
  Color of the image can be change using cv2.cvtColor() like bgr to gray, bgr to rgb, bgr to hsv etc.
  ```
  import cv2
  import numpy as np
  
  img=cv2.imread("Baby.jpg")
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  # gray contains gray version of original image
  
  kernel = np.ones((5,5),np.float32)/25
  dst = cv2.filter2D(img,-1,kernel)
  # Filter2D - To convolve kernel with an image
  
  cv2.imshow("Gray_image",gray)
  cv2.imshow("Filter",dst)
  ```
Run the Color_Filter.py to see the filtering of color.  
## 2.Blurring and Smoothing 
   There are many blurring and smoothing techniques.
* **Averaging:**
It convolves the image with normalised box filter. It takes the average of all pixels under the kernel and replace the central area with the average.
```
img=cv2.imread("Baby.jpg")
blur=cv2.blur(img,(3,3))
cv2.imshow("Blur",blur)
```
* **Gaussian Filtering:**
It uses Gaussian kernel.To create a Gaussian kernel,use cv2.getGaussianKernel().
```
img=cv2.imread("Baby.jpg")

#GaussianBlur sacrifies a lot of granularity
blur=cv2.GaussianBlur(img,(15,15),0)
cv2.imshow("Blur",blur)
```
* **Median Filtering:**
The function cv2.medianBlur() computes the median of all the pixels under the kernel window and the central pixel is replaced with this median value.
```
img=cv2.imread("Baby.jpg")
blur=cv2.medianBlur(img,3)
#Reduces the noise effectively.
cv2.imshow("Blur",blur)
```
* **Bilateral Filtering:**
It is highly effective at noise removal while preserving edges.
```
img=cv2.imread("Baby.jpg")
blur=cv2.bilateralFilter(img,9,75,75)
#Texture on the surface is gone, but edges are still preserved.
cv2.imshow("Blur",blur)
```
## 3.Thresholding:
If pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black). The function used is cv2.threshold.
Download Thresh.py and run it for thresholding.
```
#Adaptive threshold is another type of threaholding.
img = cv2.imread('Baby.jpg',0)
th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
```
## 4.Canny Edge Detection:
The first step in Edge detection is to remove the noise in the image with a 5x5 Gaussian filter.
```
img = cv2.imread('Baby.jpg',0)
blur=cv2.GaussianBlur(img,(15,15),0)
edges = cv2.Canny(blur,100,200)
```
