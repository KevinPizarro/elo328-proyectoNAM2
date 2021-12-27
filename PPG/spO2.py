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
    # print(summ) 
    total = len(color)*len(color[0])
    return summ/total

def Color_Catcher(img):
    azul = img[:,:,0]
    verde = img[:,:,1]
    rojo = img[:,:,2]
    print(Promediar_Color(azul))
    print(Promediar_Color(verde))
    print(Promediar_Color(rojo))


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