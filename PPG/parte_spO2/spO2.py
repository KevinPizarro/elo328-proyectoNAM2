import cv2
import csv
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



def Promediar_Color(color):
    summ = 0
    for i in color:
        for j in i:
            summ += j
    total = len(color)*len(color[0])
    return summ/total

def Color_Mean(img):
    azul = img[:,:,0]
    verde = img[:,:,1]
    rojo = img[:,:,2]
    P_b = Promediar_Color(azul)
    P_g = Promediar_Color(verde)
    P_r = Promediar_Color(rojo)
    return P_b, P_g, P_r

def AC_Calculation(P_b, P_g, P_r, AC_R, AC_G, AC_B, MIN_B, MIN_G, MIN_R,  MAX_B, MAX_G, MAX_R, count):
    if (P_b < MIN_B):
        MIN_B = P_b
    if (P_g < MIN_G):
        MIN_G = P_g
    if (P_r < MIN_R):
        MIN_R = P_r
    if (P_b > MAX_B):
        MAX_B = P_b
    if (P_g > MAX_G):
        MAX_G = P_g
    if (P_r > MAX_R):
        MAX_R = P_r

    if(count == 50):    
        AC_B = MAX_B - MIN_B
        AC_G = MAX_G - MIN_G
        AC_R = MAX_R - MIN_R
    
    return AC_B, AC_G, AC_R, MIN_B, MIN_G, MIN_R,  MAX_B, MAX_G, MAX_R

def DC_Calculation(P_b, P_g, P_r, DC_R, DC_G, DC_B, count):
    DC_B += P_b
    DC_G += P_g
    DC_R += P_r

    if(count == 50):
        DC_B = DC_B/50
        DC_G = DC_G/50
        DC_R = DC_R/50

    return DC_B, DC_G, DC_R
    

# funcion que recibe las componentes ac, dc y el regresor para aproximar la SpaO2 (H4.7)
def SVR_PREDICT(AC_R, AC_G, AC_B, DC_R, DC_G, DC_B, regressor):
    RoR = (AC_R/DC_R)/(AC_B/DC_B)
    SpaO2 = regressor.predict(np.array(RoR).reshape(-1,1))
    return SpaO2

# funcion que me entrena el modelo del SVR creando el modelo regressor (H4.7)
def creat_SVR_FUNCTION():
    x = []
    y = []
    f = open('./spaO2_dataset.csv', 'r')
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