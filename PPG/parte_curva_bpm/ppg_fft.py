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
import numpy as np

def actualizar_csv(csvStr=None):
    now = datetime.now()
    if csvStr is None:
        csvFileName = 'ppg_'+now.strftime("%Y-%m-%d_%I_%M_%S")
    else:
        csvFileName = csvStr+'_'+now.strftime("%Y-%m-%d_%I_%M_%S")

    headers = (u'cam_time'+','+u'cam_waveform'+','+u'cam_bpm')
    with io.open(csvFileName + '.csv', 'w', newline='') as f:
        f.write(headers)
        f.write(u'\n')
    return csvFileName

def Guardar_csv(csvFileName, data):
    with io.open(csvFileName + ".csv", "a", newline="") as f:

        row = str(data['cam_time'])+","+str(data['cam_pulseWaveform'])+","+str(data['cam_bpm'])
        f.write(row)
        f.write("\n") 

def update():
    global camData, camCurve, ptr, t, filename

    # se toma la dara
    image, signal = grabCam() ##se tiene  la imagen en escala de grises y el promedio de las intensidades del brillo de la imagen
    print('El valor de signal es: %s bpm' %signal)
    ### heartpy
    cam_bpm = camBPMData[-1] ##guarda el último valor de camBPMData
    camSig = camData - np.nanmean(camData) ## se toman los valores tomados y se le resta la media aritmetica por los 0 
    ## se hace esto para que al momento de entrar al hp.process, se muestre la diferencia entre las imagenes y eso se
    try:
        working_data, measures = hp.process(camSig, 10.0) ##ocupa la libreria de hearthpy la cual hace el procesamiento de la fft a la imagen recibida
        ###print('breathing rate is: %s bpm' %measures['bpm'])
    except BadSignalWarning:
        print("Mala señal")## en caso de que falle el hp.process()
    else:
        if(measures['bpm'] > 40 and measures['bpm'] < 120):
            cam_bpm = measures['bpm']
            ###print('breathing rate is: %s bpm' %measures['bpm'])
    ### fin HeartPy

    # vice versa
    image = image.T[:, ::-1]## da vuelta la imagen y la muestra
    img.setImage(image, autoLevels=True) ## se coloca la imagen a la variable global que se ejecuta en la GUI

    camData[:-1] = camData[1:]  # desplazar datos en la matriz una muestra a la izquierda
    camData[-1] = signal ## cambia solamente el ultimo valor de camData para el grafico

    camBPMData[:-1] = camBPMData[1:]##Lo que se hace aca es correr todo el arreglo de derecha a izquierda un espacio y copiar el ultimo en la ultima posicion.
    camBPMData[:-1] = cam_bpm ## genero un arreglo de 50 con los valores del ultimo valor de bpm de tal forma de tener como una linea constante al plotear
    ##print('El camBPMData corresponde al valor de : %s bpm' %camBPMData)
    t[:-1] = t[1:]
    t[-1] = (datetime.now() - start_time).total_seconds()


    # Los datos del paquete se guardarán en CSV.
    single_record = {}
    single_record['cam_pulseWaveform'] = camData[-1]## el último valor del arreglo.
    single_record['cam_bpm'] =cam_bpm
    single_record['cam_time'] = t[-1]
    Guardar_csv(filename, single_record)
    ## fin del csv

    ptr += 1
    camCurve.setData(camData)
    camCurve.setPos(ptr, 0)

    camBPMCurve.setData(camBPMData)
    camBPMCurve.setPos(ptr, 0)

    print(t[-1])



def grabCam():
    ret, frame = cap.read() # obtiene un fotograma de la cámara web

    # se convierten los canales solo a gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Cree un ROI redimensionable
    col,row = box.pos()
    row = int(row)
    col = int(col)
    
    x,y = box.size()
    x = int(x)
    y = int(y)

    roi = gray[row:row+y, col:col+x]
    ## fin Roi

    # Encuentra la intensidad (¿media, mediana o suma?)
    rowSum = np.sum(roi, axis=0)
    colSum = np.sum(rowSum, axis=0)
    allSum = rowSum + colSum

    intensity = np.median(np.median(allSum))
    return gray, intensity

now = datetime.now()
start_time = datetime.now()
##Qt GUI
pg.mkQApp()
win = pg.GraphicsLayoutWidget()
camPen = pg.mkPen(width=10, color='y')
win.setWindowTitle('Remote PPG')


### Acceso a la cámara
cap = cv2.VideoCapture(0)

# lee de la cámara (OpenCV)
ret, frame = cap.read() 
h = frame.shape[0] # columnas
w = frame.shape[1] # filas
aspect = h/w
## end OpenCV

#### GUI Setup
## ploteo de la imagen 
imgPlot = win.addPlot(colspan=2)
imgPlot.getViewBox().setAspectLocked(True)
win.nextRow()

## Plot for camera intensity
camPlot = win.addPlot()
camBPMPlot = win.addPlot()
win.nextRow()

# Cuadro ImageItem para mostrar datos de imagen
img = pg.ImageItem()
imgPlot.addItem(img)
imgPlot.getAxis('bottom').setStyle(showValues=False)
imgPlot.getAxis('left').setStyle(showValues=False)
imgPlot.getAxis('bottom').setPen(0,0,0)
imgPlot.getAxis('left').setPen(0,0,0)

win.show() # desplegar la pantalla
#### end GUI Setup

# frecuencia de sample, con esto por ejemplo si son 100 es que vamos a tener 100 muestras en 1 segundo
fs = 25 ## este valor fue obtenido por la funcion creada de "calcular fps de camara.py" de tal forma de no tener tantos errores 
# Inicializar
camData = np.random.normal(size=50)
camBPMData = np.zeros(50)

camPlot.getAxis('bottom').setStyle(showValues=False)
camPlot.getAxis('left').setStyle(showValues=False)

camBPMPlot.getAxis('bottom').setStyle(showValues=False)
camBPMPlot.setLabel('left','Cam BPM')

# Se usa linspace en lugar de arange debido a errores de espaciado
t = np.linspace(start=0,stop=5.0,num=50)

camCurve = camPlot.plot(t, camData, pen=camPen,name="Camera")
camPlot.setLabel('left','Cam Signal')

camBPMCurve = camBPMPlot.plot(t,camBPMData,pen=camPen,name="Cam BPM")

ptr = 0 ## para fijar posicion

## saca un frame de la camara
ret, frame = cap.read() 

# Escala de grises
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
numColumns = gray.shape[0]
numRows = gray.shape[1]
        
middleRow = int(numRows/2)
middleColumns = int(numColumns/2)

boxH = int(numRows*0.15)
boxW = int(numColumns*0.15)

box = pg.RectROI( (middleRow-boxH/2,middleColumns-boxW/2), \
    (boxH,boxW), pen=9, sideScalers=True, centered=True)
imgPlot.addItem(box)

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