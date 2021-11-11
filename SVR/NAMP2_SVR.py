# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:31:40 2021

@author: Gustavo
"""
def determinacion_threshold(frame):
    frame_YCrCb=cv.cvtColor(frame,cv.COLOR_BGR2YCR_CB)
    Y,Cr,Cb = cv.split(frame_YCrCb)
    th_Cb , step1_Cb = cv.threshold(Cb,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8) 
    img_erosion = cv.erode(step1_Cb, kernel, iterations=1) 
    img_dilation = cv.dilate(img_erosion, kernel, iterations=1)
    rows = img_dilation.shape[0]
    cols = img_dilation.shape[1]
    total = rows*cols
    r_prom = 0
    g_prom = 0
    b_prom = 0
    for i in range(0,rows):
        for j in range(0,cols):  
            if (img_dilation[i][j] == 255):
                r_prom = r_prom + frame[i][j][0]
                g_prom = g_prom + frame[i][j][1]
                b_prom = b_prom + frame[i][j][2]
    A = [r_prom/total, g_prom/total, b_prom/total]
    print(A); print("\n")
    return img_dilation            

import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
width  = cap.get(3)  # float width
height = cap.get(4)  # float height
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #intentar hacer una roi
    roi_inicio = (int(width*0.1),int(height*0.1))
    roi_fin =  (int(width*0.1+150),int(height*0.1+300))
    roi_color = (0,0,255)
    thickness = 2
    #se establece el rectangulo de la ROI
    roi=cv.rectangle(frame,roi_inicio,roi_fin,roi_color,thickness)
    #Se crea una nueva mat solo con la ROI
    roi_cropped = frame[roi_inicio[1]+thickness:roi_fin[1]-thickness,roi_inicio[0]+thickness:roi_fin[0]-thickness]
    step1 = determinacion_threshold(roi_cropped)
    # Display the resulting frame
    cv.imshow('frame rgb', frame)
    #cv.imshow('only roi', roi_cropped)
    #cv.imshow('only roi YCrCb', roi_cropped_YCrCb)
    cv.imshow('step1', step1)
    
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()