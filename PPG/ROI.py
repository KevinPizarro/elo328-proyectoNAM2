import cv2
import numpy as np
import math
# import face_recognition


alpha = 1.0
beta = 20
gamma = 2.0

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
        print("Image already bright enough")
        print(ratio)
        return cv2.convertScaleAbs(src, alpha = (-1/ratio)*2, beta = 0)

    else:
        src = suavizarRuido(src, 2,2)
        return cv2.convertScaleAbs(src, alpha = 1 / ratio, beta = 0)


## Funcion que detecta el frame de la imagen y la procesa, para detectar el rostro, posteriormente la frente y la boca.
def detect():
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
    mouth_cascade = cv2.CascadeClassifier('PPG\mouth.xml')
    
    camera = cv2.VideoCapture(0)
    detection = 0
    fx = 0
    fx2 = 0
    
    while(True):
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
            for (ex,ey,ew,eh) in eyes:
                if abs(x-x+ex+ew) > w*0.51:
                    fx2 = x+ex+ew
                else:
                    fx = x+ex
                cv2.rectangle(frame, (fx,y), (fx2,y+ey), (0,255,0), 2)
            
            ## Se aplica el reconociento de boca dentro del cuadro del rostro.
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.1, 20)
            ## Se elimina el reconocimiento de bocas sobre la mitad superior de la cara.
            for (mx,my,mw,mh) in mouth:
                if y+my > y + h/2:
                    cv2.rectangle(frame, (x+mx,y+my), (x+mx+mw,y+my+mh), (255,255,0), 2)

        ## Si no se reconocen caras, se sube el brillo.
        if ( len(faces) > 0 ):
            cols, rows= faces.shape
            faces = autoAdjustBrightness(faces, cols, rows)
            detection = 0


        cv2.imshow('camera', frame)
        if cv2.waitKey(int(1000/12)) & 0xff == ord("q"):
            break


    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()