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
