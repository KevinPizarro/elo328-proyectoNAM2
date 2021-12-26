import cv2
import numpy as np
import math
# import face_recognition


alpha = 1.0
beta = 20
gamma = 2.0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('PPG\mouth.xml')
camera = cv2.VideoCapture(0)
detection = 0
forehead_img = 0
mouth_img = 0 

## Funcion para suavizar ruido de la imagen.
def suavizarRuido(src, k, k2):
    return cv2.blur(src,(k,k2),)

## Funcion para corregir gamma automaticamente, dependiendo del gamma de la imagen.
def autoGammaCorrection(src):
    # convert img to HSV
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.3
    mean = np.mean(val)
    gamma = math.log(mid*255)/math.log(mean)
    # print(gamma)

    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

    return img_gamma2


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


## Funcion que detecta el frame de la imagen y la procesa, para detectar el rostro, posteriormente la frente y la boca.
def detect():
    while(True):
        mouth_in, forehead_in = grabCam(detection,forehead_img, mouth_img)
        print(mouth_in,forehead_in)
        if cv2.waitKey(int(1000/12)) & 0xff == ord("q"):
            break
           
    camera.release()
    cv2.destroyAllWindows()
    
def grabCam(detection,forehead_img, mouth_img):
    ret, frame = camera.read()
        
    # frame = autoGammaCorrection(frame)

    if (detection == 0 ):
        cols, rows, _ = frame.shape
        frame = autoAdjustBrightness(frame, cols, rows)

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
            if y+my > y + h/2:
                cv2.rectangle(frame, (x+mx,y+my), (x+mx+mw,y+my+mh), (255,255,0), 2)
                mouth_img = frame[y+my:y+my+mh, x+mx:x+mx+mw].copy() 
    ## Si no se reconocen caras, se sube el brillo.
    if ( len(faces) > 0 ):
        cols, rows= faces.shape
        faces = autoAdjustBrightness(faces, cols, rows)
        detection = 0

    mouth_intensity, forehead_intensity = 0,0

    if(type(mouth_img) != int):
        mouth_gray= cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)

        mouth_rowSum = np.sum(mouth_gray, axis=0)
        mouth_colSum = np.sum(mouth_rowSum, axis=0)
        mouth_allSum = mouth_rowSum + mouth_colSum
        mouth_intensity = np.median(np.median(mouth_allSum))

    if(type(forehead_img) != int):
        forehead_gray = cv2.cvtColor(forehead_img, cv2.COLOR_BGR2GRAY)

        forehead_rowSum = np.sum(forehead_gray, axis=0)
        forehead_colSum = np.sum(forehead_rowSum, axis=0)
        forehead_allSum = forehead_rowSum + forehead_colSum
        forehead_intensity = np.median(np.median(forehead_allSum))

    cv2.imshow('camera', frame)
    
    return mouth_intensity, forehead_intensity

if __name__ == "__main__":
    detect()