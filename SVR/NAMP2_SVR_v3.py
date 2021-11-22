# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:31:40 2021

@author: Gustavo Silva, Kevin Pizarro
@objective: Make the code to predict de SPO2 and respiration curve using rPPG and SVG 
"""

import numpy as np
import cv2 as cv
import os
import csv

# Directorio donde se encuentra este Programa idealmente
# Modificar para cada usuario
PATH = "." 

# Numero de la camara que esta utilizando
CAM_NUMBER = 0


# Crear una mascara de la mano (reconocimiento de la piel), en una region estatica.
# Si el pixel pertenece a la RoI entonces lo añado al calculo del promedio por canal 
def determinacion_threshold(frame):
    frame_YCrCb=cv.cvtColor(frame,cv.COLOR_BGR2YCR_CB)
    Y,Cr,Cb = cv.split(frame_YCrCb)
    th_Cb , step1_Cb = cv.threshold(Cr,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
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
    return img_dilation , A

# Recibir la matriz de promedio del color RGB del pixel actual y los anteriores
# concatenar estas matrices formando mi matriz A actualizada
# formar las curvas temporales de R G B 
def rPPG_extraction(A):
    #el valor de 100 fue arbitrario por nosotros al notar ruido de medicion
    #cuando no esta la mano
    if ( os.path.exists(PATH+'/Matris_A.csv')): #si existe
        f = open(PATH+'/Matris_A.csv', 'a+')
        writer = csv.writer(f)
        writer.writerow( A )
        f.close()
    if ( os.path.exists(PATH+'/Matris_A.csv') ): #si existe    
        f = open(PATH+'/Matris_A.csv', 'r')
        reader = csv.reader(f)
        i=0; #ventana de i datos
        buffer_R = np.zeros(50)
        buffer_G = np.zeros(50)
        buffer_B = np.zeros(50)
        RoR_max=4
        RoR_min=1
        for line in reader:
            if ( line != [] ):
                #print("iteracion ",i)
                R,G,B = line
                buffer_R[i%50] = float(R)
                buffer_G[i%50] = float(G)
                buffer_B[i%50] = float(B)
                if (i%50 == 0):                    
                    AC_R = np.amax(buffer_R) - np.amin(buffer_R)
                    #AC_G = np.amax(buffer_G) - np.amin(buffer_G)
                    AC_B = np.amax(buffer_B) - np.amin(buffer_B)
                    DC_R = np.mean(buffer_R)
                    #DC_G = np.mean(buffer_G)
                    DC_B = np.mean(buffer_B)
                    RoR = (AC_R/DC_R)/(AC_B/DC_B)
                    m = (70-100)/(RoR_max-RoR_min)
                    n = 100 - m*RoR_min
                    SpaO2 = n + m*RoR
                    print("el valor de SPA_O2 es: ", SpaO2,"%\n")
                i=i+1
        f.close()
    return 1

#-----------------------------------------------------------------------------#
#----------------------Comienzo---de---programa-------------------------------#
# crear archivo .csv y en el caso que existe lo va a sobre-escribir
f = open(PATH+'/Matris_A.csv', 'w+')
f.close()
#obtener video de la camara
cap = cv.VideoCapture(CAM_NUMBER)
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
    step1 , A = determinacion_threshold(roi_cropped)
    #print(A); print("\n")
    rPPG_extraction(A)    
    # Display the resulting frame
    cv.imshow('frame rgb', frame)
    cv.imshow('step1', step1)
    # end teh program with the event keypressed-q
    if ( ( cv.waitKey(1) == ord('q') ) or ( cv.waitKey(1) == ord('Q') ) ):
        break

# When everything done...
f.close()
cap.release()
cv.destroyAllWindows()
#-----------------------------------------------------------------------------#
#----------------------------FIN---de---programa------------------------------#