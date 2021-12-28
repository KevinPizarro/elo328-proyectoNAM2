##Código realizado por:
##Victor Cortés
##Johanny Espinoza

# OpenCV para acceso a la cámara y lectura de fotogramas
import cv2
# Paquete de análisis de frecuencia cardíaca
import heartpy as hp
from heartpy.exceptions import BadSignalWarning
# Para guardar archivos
from datetime import datetime
import io
# GUI y herramientas de trazado
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import spO2
import sys 
import math
# import face_recognition

alpha = 1.0
beta = 20
gamma = 2.0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('./mouth.xml')
detector = FaceDetector()
detection = 0
count = 0
FDC_B, FDC_G, FDC_R = 0, 0, 0
FAC_B, FAC_G, FAC_R = 0, 0, 0
FMAX_B, FMAX_G, FMAX_R = 0, 0, 0
FMIN_B, FMIN_G, FMIN_R = 255, 255, 255

MDC_B, MDC_G, MDC_R = 0, 0, 0
MAC_B, MAC_G, MAC_R = 0, 0, 0
MMAX_B, MMAX_G, MMAX_R = 0, 0, 0
MMIN_B, MMIN_G, MMIN_R = 255, 255, 255

regressor = spO2.creat_SVR_FUNCTION()

# forehead_img = 0
# mouth_img = 0 

## Funcion para suavizar ruido de la imagen.
def suavizarRuido(src, k, k2):
    return cv2.blur(src,(k,k2),)


##  Funcion para ajustar el Brillo de forma automatica, dependeindo del brillo actual de la imagen.
def autoAdjustBrightness(src, cols, rows):

    brightness = np.sum(src) / (255 * cols * rows)
    
    minimum_brightness = 0.66
    # bright_img = cv2.convertScaleAbs(img, alpha = alpha, beta = 255 * (1 - alpha))

    ratio = brightness / minimum_brightness
    if ratio >= 1:
        # print("Image already bright enough")
        # print(ratio)
        return cv2.convertScaleAbs(src, alpha = (-1/ratio)*2, beta = 0)

    else:
        src = suavizarRuido(src, 2,2)
        return cv2.convertScaleAbs(src, alpha = 1 / ratio, beta = 0)

def actualizar_csv(csvStr=None):
    now = datetime.now()
    if csvStr is None:
        csvFileName = 'ppg_'+now.strftime("%Y-%m-%d_%I_%M_%S")
    else:
        csvFileName = csvStr+'_'+now.strftime("%Y-%m-%d_%I_%M_%S")

    headers = (u'cam_time'+','+u'cam_pulseWaveform_forehead'+','+u'cam_bpm_forehead'+','+u'cam_pulseWaveform_mouth'+','+u'cam_bpm_mouth')
    with io.open(csvFileName + '.csv', 'w', newline='') as f:
        f.write(headers)
        f.write(u'\n')
    return csvFileName

def Guardar_csv(csvFileName, data):
    with io.open(csvFileName + ".csv", "a", newline="") as f:
        row = str(data['cam_time'])+","+str(data['cam_pulseWaveform_forehead'])+","+str(data['cam_bpm_forehead'])+","+str(data['cam_pulseWaveform_mouth'])+","+str(data['cam_bpm_mouth'])
        f.write(row)
        f.write("\n") 

def update():
    global camData, camData2, camCurve, camCurve2, ptr, t, filename, count
    global FDC_B, FDC_G, FDC_R, FAC_B, FAC_G, FAC_R, FMAX_B, FMAX_G, FMAX_R, FMIN_B, FMIN_G, FMIN_R
    global MDC_B, MDC_G, MDC_R, MAC_B, MAC_G, MAC_R, MMAX_B, MMAX_G, MMAX_R, MMIN_B, MMIN_G, MMIN_R 
    global regressor 

    # se toma la dara
    image, signal, signal2, forehead_img, mouth_img = grabCam() ##se tiene  la imagen en escala de grises y el promedio de las intensidades del brillo de la imagen
    #image, signal, signal2, forehead_img, mouth_img = grabCam2() ##se tiene  la imagen en escala de grises y el promedio de las intensidades del brillo de la imagen
    count += 1

    ## Se empieza el algoritmo para calcular la saturacion de oxigeno
    if (len(forehead_img) > 0):
        FP_b, FP_g, FP_r = spO2.Color_Mean(forehead_img)
        FAC_R, FAC_G, FAC_B, FMIN_B, FMIN_G, FMIN_R, FMAX_B, FMAX_G, FMAX_R = spO2.AC_Calculation(FP_b, FP_g, FP_r, FAC_R, FAC_G, FAC_B, FMIN_B, FMIN_G, FMIN_R, FMAX_B, FMAX_G, FMAX_R, count)
        FDC_R, FDC_G, FDC_B = spO2.DC_Calculation(FP_b, FP_g, FP_r, FDC_R, FDC_G, FDC_B, count)

    if (len(mouth_img) > 0):
        MP_b, MP_g, MP_r = spO2.Color_Mean(mouth_img)
        MAC_R, MAC_G, MAC_B, MMIN_B, MMIN_G, MMIN_R, MMAX_B, MMAX_G, MMAX_R = spO2.AC_Calculation(MP_b, MP_g, MP_r, MAC_R, MAC_G, MAC_B, MMIN_B, MMIN_G, MMIN_R, MMAX_B, MMAX_G, MMAX_R, count)
        MDC_R, MDC_G, MDC_B = spO2.DC_Calculation(MP_b, MP_g, MP_r, MDC_R, MDC_G, MDC_B, count)
    
    if (count == 50):
        
        # print(FAC_R, FAC_G, FAC_B, FDC_R, FDC_G, FDC_B, regressor)
        # print(MAC_R, MAC_G, MAC_B, MDC_R, MDC_G, MDC_B, regressor)
        if(FAC_R > 0):
            Fsat = spO2.SVR_PREDICT(FAC_R, FAC_G, FAC_B, FDC_R, FDC_G, FDC_B, regressor)
            #print('La saturacion de Oxigeno en la sangre segun la medicion de la frente es: ', Fsat)


        if(MAC_R > 0):
            Msat = spO2.SVR_PREDICT(MAC_R, MAC_G, MAC_B, MDC_R, MDC_G, MDC_B, regressor)
            #print('La saturacion de Oxigeno en la sangre segun la medicion de la boca es: ', Msat)

        count = 0
    

    #print('El valor de signal es: %s bpm' %signal)
    
    ### heartpy
    cam_bpm = camBPMData[-1] ##guarda el último valor de camBPMData
    camSig = camData - np.nanmean(camData) ## se toman los valores tomados y se le resta la media aritmetica por los 0 
    ## se hace esto para que al momento de entrar al hp.process, se muestre la diferencia entre las imagenes y eso se
    cam_bpm2 = camBPMData2[-1] ##guarda el último valor de camBPMData
    camSig2 = camData2 - np.nanmean(camData2) ## se toman los valores tomados y se le resta la media aritmetica por los 0
    if(signal == 0):
        signal = camData[-1]
    
    else:
        signal = signal

    if(signal2 == 0):
        signal2 = camData2[-1]
    
    else:
        signal2 = signal2

    try:
        working_data, measures = hp.process(camSig, 10.0) ##ocupa la libreria de hearthpy la cual hace el procesamiento de la fft a la imagen recibida
        ###print('breathing rate is: %s bpm' %measures['bpm'])
    except BadSignalWarning:
        pass
        #print("Mala señal")## en caso de que falle el hp.process()
    else:
        if(measures['bpm'] > 40 and measures['bpm'] < 120):
            cam_bpm = measures['bpm']
            ###print('breathing rate is: %s bpm' %measures['bpm'])
    ### fin HeartPy

    try:
        working_data, measures2 = hp.process(camSig2, 10.0) 
    except BadSignalWarning:
        pass    
        #print("Mala señal")
    else:
        if(measures2['bpm'] > 40 and measures2['bpm'] < 120):
            cam_bpm2 = measures2['bpm']

    # vice versa
    #image = image.T[:, ::-1]## da vuelta la imagen y la muestra
    #img.setImage(image, autoLevels=True) ## se coloca la imagen a la variable global que se ejecuta en la GUI

    camData[:-1] = camData[1:]  # desplazar datos en la matriz una muestra a la izquierda
    camData[-1] = signal ## cambia solamente el ultimo v    alor de camData para el grafico

    camBPMData[:-1] = camBPMData[1:]##Lo que se hace aca es correr todo el arreglo de derecha a izquierda un espacio y copiar el ultimo en la ultima posicion.
    camBPMData[:-1] = cam_bpm ## genero un arreglo de 50 con los valores del ultimo valor de bpm de tal forma de tener como una linea constante al plotear

    camData2[:-1] = camData2[1:]  # desplazar datos en la matriz una muestra a la izquierda
    camData2[-1] = signal2 ## cambia solamente el ultimo v    alor de camData para el grafico

    camBPMData2[:-1] = camBPMData2[1:]##Lo que se hace aca es correr todo el arreglo de derecha a izquierda un espacio y copiar el ultimo en la ultima posicion.
    camBPMData2[:-1] = cam_bpm2 ## genero un arreglo de 50 con los valores del ultimo valor de bpm de tal forma de tener como una linea constante al plotear

    ##print('El camBPMData corresponde al valor de : %s bpm' %camBPMData)
    t[:-1] = t[1:]
    t[-1] = (datetime.now() - start_time).total_seconds()


    # Los datos del paquete se guardarán en CSV.
    single_record = {}
    single_record['cam_time'] = t[-1]
    single_record['cam_pulseWaveform_forehead'] = camData[-1]## el último valor del arreglo.
    single_record['cam_bpm_forehead'] =cam_bpm
    single_record['cam_pulseWaveform_mouth'] = camData2[-1]## el último valor del arreglo.
    single_record['cam_bpm_mouth'] =cam_bpm2
    Guardar_csv(filename, single_record)
    ## fin del csv

    ptr += 1
    camCurve.setData(camData)
    camCurve.setPos(ptr, 0)

    camCurve2.setData(camData2)
    camCurve2.setPos(ptr, 0)

    camBPMCurve.setData(camBPMData)
    camBPMCurve.setPos(ptr, 0)
    camBPMCurve2.setData(camBPMData2)
    camBPMCurve2.setPos(ptr, 0)

    #print(t[-1])



def grabCam():
    ret, frame = cap.read()
    forehead_img, mouth_img = [],[]
#   frame = autoGammaCorrection(frame)
#    if (detection == 0 ):
#       cols, rows, _ = frame.shape
#       frame = autoAdjustBrightness(frame, cols, rows)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## Se aplica el reconocimiento facial.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        detection = 1
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        # img = cv2.circle(frame, (int(x+abs(x-w)),int(y+abs(y-h))), int(abs(y-h)) , (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        ## Se aplica el reconocimiento de ojos, dentro del cuadro del rostro.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
        ## Se dibuja el rectangulo, arriba de los parametros de los ojos.
        ex,ey,ew,eh = 99999,0,0,0
        izq = 0
        for (exx,eyy,eww,ehh) in eyes:
            if y+eyy < y + h/2 :
                if ex > exx :
                    ex = exx
                if izq < exx+eww :
                    izq = exx+eww
                    ey = eyy

        cv2.rectangle(frame, (x+ex,y), (x+izq,y+ey), (0,255,0), 2)
        forehead_img = frame[y:y+ey, x+ex:x+izq].copy() 
        
        ## Se aplica el reconociento de boca dentro del cuadro del rostro.
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.1, 20)
        ## Se elimina el reconocimiento de bocas sobre la mitad superior de la cara.
        for (mx,my,mw,mh) in mouth:
            if y+my > y + 2*(h/3):
                cv2.rectangle(frame, (x+mx,y+my), (x+mx+mw,y+my+mh), (255,255,0), 2)
                mouth_img = frame[y+my:y+my+mh, x+mx:x+mx+mw].copy() 
    ## Si no se reconocen caras, se sube el brillo.
#    if ( len(faces) > 0 ):
#       cols, rows= faces.shape
#        faces = autoAdjustBrightness(faces, cols, rows)
#        detection = 0

    mouth_intensity, forehead_intensity = 0,0

    if(len(mouth_img) > 0):
        mouth_gray= cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)

        mouth_rowSum = np.sum(mouth_gray, axis=0)
        mouth_colSum = np.sum(mouth_rowSum, axis=0)
        mouth_allSum = mouth_rowSum + mouth_colSum
        mouth_intensity = np.median(np.median(mouth_allSum))

    if(len(forehead_img) > 0):
        forehead_gray = cv2.cvtColor(forehead_img, cv2.COLOR_BGR2GRAY)

        forehead_rowSum = np.sum(forehead_gray, axis=0)
        forehead_colSum = np.sum(forehead_rowSum, axis=0)
        forehead_allSum = forehead_rowSum + forehead_colSum 
        forehead_intensity = np.median(np.median(forehead_allSum))

    cv2.imshow('camera', frame)
    #print("los valores para mouth y forehead son: %f y %f respectivamente" %(mouth_intensity, forehead_intensity))
    return frame, forehead_intensity, mouth_intensity, forehead_img, mouth_img

def grabCam2():
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    mouth_intensity, forehead_intensity = 0,0
    forehead_img, mouth_img = [],[]

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        # print(bboxs)
        center = bboxs[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
    
        eyes = eye_cascade.detectMultiScale(img, 1.03, 5, 0, (40, 40))
        x = bboxs[0]['bbox'][0]
        y = bboxs[0]['bbox'][1]
        w = bboxs[0]['bbox'][2]
        h = bboxs[0]['bbox'][3]
        cx = bboxs[0]['center'][0]
        cy = bboxs[0]['center'][1]

        ex,ey,ew,eh = 99999,0,0,0
        izq = 0
        for (exx,eyy,eww,ehh) in eyes:
            if eyy < cy :
                if eyy > y :
                    if ex > exx :
                        ex = exx
                    if izq < exx+eww :
                        izq = exx+eww
                        ey = eyy

            # cv2.rectangle(img, (exx,eyy), (exx+eww,eyy+ehh), (0,255,0), 2)

        if (izq > 0):
            cv2.rectangle(img, (ex,y), (izq,ey), (0,255,0), 2)
            forehead_img = img[y:y+ey, x+ex:x+izq].copy() 
        
        # Se aplica el reconociento de boca dentro del cuadro del rostro.
        mouth = mouth_cascade.detectMultiScale(img, 1.1, 20)
        ## Se elimina el reconocimiento de bocas sobre la mitad superior de la cara.
        for (mx,my,mw,mh) in mouth:
            if my > cy:
                if my+mh < y+h:
                    cv2.rectangle(img, (mx,my), (mx+mw,my+mh), (255,255,0), 2)
                    mouth_img = img[my:my+mh, mx:mx+mw].copy() 


    if(len(mouth_img) > 0):
        mouth_gray= cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)

        mouth_rowSum = np.sum(mouth_gray, axis=0)
        mouth_colSum = np.sum(mouth_rowSum, axis=0)
        mouth_allSum = mouth_rowSum + mouth_colSum
        mouth_intensity = np.median(np.median(mouth_allSum))

    if(len(forehead_img) > 0):
        forehead_gray = cv2.cvtColor(forehead_img, cv2.COLOR_BGR2GRAY)

        forehead_rowSum = np.sum(forehead_gray, axis=0)
        forehead_colSum = np.sum(forehead_rowSum, axis=0)
        forehead_allSum = forehead_rowSum + forehead_colSum
        forehead_intensity = np.median(np.median(forehead_allSum))

    cv2.imshow("Image", img)
    #print("los valores para mouth y forehead son: %f y %f respectivamente" %(mouth_intensity, forehead_intensity))
    return img, forehead_intensity, mouth_intensity, forehead_img, mouth_img



now = datetime.now()
start_time = datetime.now()
##Qt GUI 
pg.mkQApp()
win = pg.GraphicsLayoutWidget()
camPen = pg.mkPen(width=10, color='y')
win.setWindowTitle('Remote PPG')
detection = 0

### Acceso a la cámara
if (sys.argv[1]=="0"):
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(sys.argv[1])



#### GUI Setup
## ploteo de la imagen 
imgPlot = win.addPlot(colspan=2)
#imgPlot.getViewBox().setAspectLocked(True)
#win.nextRow()

## Plot for camera intensity
camPlot = win.addPlot()
camPlot2 = win.addPlot()
camBPMPlot = win.addPlot()
camBPMPlot2 = win.addPlot()
win.nextRow()

# Cuadro ImageItem para mostrar datos de imagen
#img = pg.ImageItem()
#imgPlot.addItem(img)
imgPlot.getAxis('top').setStyle(showValues=False)
imgPlot.getAxis('right').setStyle(showValues=False)
imgPlot.getAxis('bottom').setStyle(showValues=False)
imgPlot.getAxis('left').setStyle(showValues=False)
imgPlot.getAxis('bottom').setPen(0,0,0)
imgPlot.getAxis('left').setPen(0,0,0)
imgPlot.getAxis('top').setPen(0,0,0)
imgPlot.getAxis('right').setPen(0,0,0)

win.show() # desplegar la pantalla
#### end GUI Setup

# frecuencia de sample, con esto por ejemplo si son 100 es que vamos a tener 100 muestras en 1 segundo
fs = 25 ## este valor fue obtenido por la funcion creada de "calcular fps de camara.py" de tal forma de no tener tantos errores 
# Inicializar
camData = np.random.normal(size=50)
camBPMData = np.zeros(50)

camData2 = np.random.normal(size=50)
camBPMData2 = np.zeros(50)

camPlot.getAxis('bottom').setStyle(showValues=False)
camPlot.getAxis('left').setStyle(showValues=False)
camPlot2.getAxis('top').setStyle(showValues=False)
camPlot2.getAxis('right').setStyle(showValues=False)

camBPMPlot.getAxis('bottom').setStyle(showValues=False)
camBPMPlot.setLabel('left','Cam BPM forehead')
camBPMPlot2.getAxis('top').setStyle(showValues=False)
camBPMPlot2.setLabel('right','Cam BPM mouth')

# Se usa linspace en lugar de arange debido a errores de espaciado
t = np.linspace(start=0, stop=5.0, num=50)

camCurve = camPlot.plot(t, camData, pen=camPen,name="Camera")
camPlot.setLabel('left','Cam Signal')
camCurve2 = camPlot2.plot(t, camData2, pen=camPen,name="Camera")
camPlot2.setLabel('right','Cam Signal')


camBPMCurve = camBPMPlot.plot(t,camBPMData,pen=camPen,name="Cam BPM forehead")
camBPMCurve2 = camBPMPlot2.plot(t,camBPMData2,pen=camPen,name="Cam BPM mouth")


ptr = 0 ## para fijar posicion

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
tickTime = 1000/fs # cuántos milisegundos esperar.
timer.start(tickTime)## empieza el tick timer

## Configurar archivo CSV
filename=actualizar_csv()

## Inicie el bucle de eventos Qt
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()