import cv2
import numpy as np
import spO2
import sys

# import face_recognition


alpha = 1.0
beta = 20
gamma = 2.0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('./mouth.xml')
# camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture('./Video 004a V.Trujillo.mp4')
# camera = cv2.VideoCapture('./Video 009c M.CUBILLOS.mp4')
# camera = cv2.VideoCapture('./Video 014a E.Trujillo.mp4')

detection = 0

### Acceso a la cámara
if (sys.argv[1]=="0"):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(sys.argv[1])


##  Funcion para ajustar el Brillo de forma automatica, dependeindo del brillo actual de la imagen.
def autoAdjustBrightness(src, cols, rows):

    brightness = np.sum(src) / (255 * cols * rows)
    
    minimum_brightness = 0.66

    ratio = brightness / minimum_brightness
    if ratio >= 1:
        return cv2.convertScaleAbs(src, alpha = (-1/ratio)*2, beta = 0)

    else:
        return cv2.convertScaleAbs(src, alpha = 1 / ratio, beta = 0)


## Funcion que detecta el frame de la imagen y la procesa, para detectar el rostro, posteriormente la frente y la boca.
def detect():
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

    while(True):
        img, mouth_img, forehead_img = grabCam(detection)
        count += 1
        if (len(forehead_img) > 0):
            FP_b, FP_g, FP_r = spO2.Color_Mean(forehead_img)
            FAC_R, FAC_G, FAC_B, FMIN_B, FMIN_G, FMIN_R, FMAX_B, FMAX_G, FMAX_R = spO2.AC_Calculation(FP_b, FP_g, FP_r, FAC_R, FAC_G, FAC_B, FMIN_B, FMIN_G, FMIN_R, FMAX_B, FMAX_G, FMAX_R, count)
            FDC_R, FDC_G, FDC_B = spO2.DC_Calculation(FP_b, FP_g, FP_r, FDC_R, FDC_G, FDC_B, count)

        if (len(mouth_img) > 0):
            MP_b, MP_g, MP_r = spO2.Color_Mean(mouth_img)
            MAC_R, MAC_G, MAC_B, MMIN_B, MMIN_G, MMIN_R, MMAX_B, MMAX_G, MMAX_R = spO2.AC_Calculation(MP_b, MP_g, MP_r, MAC_R, MAC_G, MAC_B, MMIN_B, MMIN_G, MMIN_R, MMAX_B, MMAX_G, MMAX_R, count)
            MDC_R, MDC_G, MDC_B = spO2.DC_Calculation(MP_b, MP_g, MP_r, MDC_R, MDC_G, MDC_B, count)
        
        if (count == 50):
            
            if(FAC_R > 0):
                Fsat = spO2.SVR_PREDICT(FAC_R, FAC_G, FAC_B, FDC_R, FDC_G, FDC_B, regressor)
                print('spO2 Frente: ',Fsat)

            if(MAC_R > 0):
                Msat = spO2.SVR_PREDICT(MAC_R, MAC_G, MAC_B, MDC_R, MDC_G, MDC_B, regressor)
                print('spO2 Boca: ',Msat)

            count = 0

        if cv2.waitKey(int(1000/12)) & 0xff == ord("q"):
            break
           
    camera.release()
    cv2.destroyAllWindows()
    
def grabCam(detection):
    ret, frame = camera.read()
    forehead_img = []
    mouth_img = []

    if (detection == 0 ):
        cols, rows, _ = frame.shape
        frame = autoAdjustBrightness(frame, cols, rows)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## Se aplica el reconocimiento facial.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        detection = 1
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
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
    if ( len(faces) > 0 ):
        cols, rows= faces.shape
        faces = autoAdjustBrightness(faces, cols, rows)
        detection = 0


    cv2.imshow('camera', frame)
    
    return frame, mouth_img, forehead_img



if __name__ == "__main__":
    detect()
