# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:31:40 2021

@author: Gustavo Silva, Kevin Pizarro
@objective: Make the code to predict de SPO2 and respiration curve using rPPG and SVG 
"""
# import sys
#from scipy import signal
from scipy.signal import butter, lfilter
import heartpy as hp
import numpy as np
#import matplotlib.pyplot as plt
import cv2 as cv
import os
import csv
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Directorio donde se encuentra este Programa idealmente
# Modificar para cada usuario
PATH = "." 

# Numero de la camara que esta utilizando
CAM_NUMBER = 0


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# se crea la señal de pulso H
def POS2(R_v,G_v,B_v):
    l = 30
    mean_rgb= np.array([R_v, G_v, B_v])
    H = np.zeros(mean_rgb.shape[1])
    #print("H shape", H.shape)
    for t in range(0, 50):
        if(t+l-1>0):
            #t = 0
            # Step 1: Spatial averaging
            C = mean_rgb[:t+l-1,:]
            #C = mean_rgb.T
            #print("C shape", C.shape)
            #print("t={0},t+l={1}".format(t,t+l))
            
            #Step 2 : Temporal normalization
            mean_color = np.mean(C, axis=1)
            #print("Mean color", mean_color)
            
            diag_mean_color = np.diag(mean_color)
            #print("Diagonal",diag_mean_color)
            
            diag_mean_color_inv = np.linalg.pinv(diag_mean_color)
            #print("Inverse",diag_mean_color_inv)
            
            Cn = (np.matmul(diag_mean_color_inv,C))
            #print("Temporal normalization", Cn)
            #print("Cn shape", Cn.shape)
        
            #Step 3: 
            projection_matrix = np.array([[0,1,-1],[-2,1,1]])
            #print("projection_matrix shape", projection_matrix.shape)
            S = np.matmul(projection_matrix,Cn)
            #print("S matrix",S)
            #print("S shape", S.shape)
    
            #Step 4:
            #2D signal to 1D signal
            std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
            #print("std shape", std.shape)
            #print("std",std)
            P = np.matmul(std,S)
            #print("P shape", P.shape)
            #print("P",P)
    
            #Step 5: Overlap-Adding
            H = H +  (P-np.mean(P))/np.std(P)
    # Setting standard filter requirements.
    order = 8
    fs = 30       
    cutoff = 10
    y = butter_lowpass_filter(H, cutoff, fs, order)
    # figure=plt.figure()
    # figure.clear()
    # plt.plot(y)
    working_data, measures = hp.process(y, fs, report_time=False) 
    BPM = -1
    if( not np.isnan(measures['breathingrate']) ):
        BPM = measures['breathingrate']*60
        #print('breathing rate is: %s Hz\n' %measures['breathingrate'])
    return BPM

# funcion que me entrena el modelo del SVR creando el modelo regressor
def creat_SVR_FUNCTION():
    x = []
    y = []
    f = open(PATH+'/spaO2_dataset.csv', 'r')
    reader = csv.reader(f)
    for line in reader:
        if ( line != [] ):
            line = line[0].split()
            c1 = float(line[0])
            c2 = float(line[1])
            x.append(c1)
            y.append(c2)
    f.close()
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    regressor = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regressor.fit(x, y)
    return regressor
# funcion que recibe las componentes ac, dc y el regresor para aproximar la SpaO2
def SVR_PREDICT(AC_R, AC_G, AC_B, DC_R, DC_G, DC_B, regressor):
    RoR = (AC_R/DC_R)/(AC_B/DC_B)
    SpaO2 = regressor.predict(np.array(RoR).reshape(-1,1))
    return SpaO2

# Crear una mascara de la mano (reconocimiento de la piel), en una region estatica.
# Si el pixel pertenece a la RoI entonces lo añado al calculo del promedio por canal 
def determinacion_threshold(frame):
    frame_YCrCb=cv.cvtColor(frame,cv.COLOR_BGR2YCR_CB)
    Y,Cr,Cb = cv.split(frame_YCrCb)
    th_Cr , step1_Cr = cv.threshold(Cr,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8) 
    img_erosion = cv.erode(step1_Cr, kernel, iterations=1) 
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
def estimated_SPAO2(A, regressor):
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
        for line in reader:
            if ( line != [] ):
                R,G,B = line
                buffer_R[i%50] = float(R)
                buffer_G[i%50] = float(G)
                buffer_B[i%50] = float(B)
                if (i%50 == 0 and i!= 0):          
                    #print("entre con 100 datos")
                    #print("\n\nbuffer_R=\n",buffer_R)
                    AC_R = np.amax(buffer_R) - np.amin(buffer_R)
                    AC_G = np.amax(buffer_G) - np.amin(buffer_G)
                    AC_B = np.amax(buffer_B) - np.amin(buffer_B)
                    DC_R = np.mean(buffer_R)
                    DC_G = np.mean(buffer_G)
                    DC_B = np.mean(buffer_B)
                    SpaO2 = SVR_PREDICT(AC_R, AC_G, AC_B, DC_R, DC_G, DC_B, regressor)[0]
                    if not (np.isnan(buffer_R).any() or np.isnan(buffer_G).any() or np.isnan(buffer_B).any()):
                        BPM = POS2(buffer_R,buffer_G,buffer_B)
                        if(BPM != -1 ):
                            print("SPA_O2 = ", SpaO2,"%")
                            print("breathing rate  = ", BPM, "BPM\n")    
                i=i+1
        f.close()
    return 1


#-----------------------------------------------------------------------------#
#----------------------Comienzo---de---programa-------------------------------#
## Entrenar la svr
regressor = creat_SVR_FUNCTION()
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
    roi_inicio = (int(width*0.1),int(height*0.1+150))
    roi_fin =  (int(width*0.1+200),int(height*0.1+350))
    roi_color = (0,0,255)
    thickness = 2
    #se establece el rectangulo de la ROI
    roi=cv.rectangle(frame,roi_inicio,roi_fin,roi_color,thickness)
    #Se crea una nueva mat solo con la ROI
    roi_cropped = frame[roi_inicio[1]+thickness:roi_fin[1]-thickness,roi_inicio[0]+thickness:roi_fin[0]-thickness]
    step1 , A = determinacion_threshold(roi_cropped)
    #print(A); print("\n")
    estimated_SPAO2(A,regressor)    
    # Display the resulting frame
    cv.imshow('frame rgb', frame)
    cv.imshow('step1', step1)
    # end teh program with the event keypressed-q
    if ( cv.waitKey(1) == ord('q') ):
        break

# When everything done...
f.close()
cap.release()
cv.destroyAllWindows()
#-----------------------------------------------------------------------------#
#----------------------------FIN---de---programa------------------------------#