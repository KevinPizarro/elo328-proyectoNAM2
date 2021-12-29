from cvzone.FaceDetectionModule import FaceDetector
import cv2
import numpy as np
import spO2 
import sys

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('./Video 004a V.Trujillo.mp4')
# cap = cv2.VideoCapture('./Video 009c M.CUBILLOS.mp4')
# cap = cv2.VideoCapture('./Video 014a E.Trujillo.mp4')
# cap = cv2.VideoCapture('./Video 003 G.Trujillo.MOV')
### Acceso a la cámara
if (sys.argv[1]=="0"):
    capture = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(sys.argv[1])




detector = FaceDetector()
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

mouth_cascade = cv2.CascadeClassifier('./mouth.xml')
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


def grabCam():
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    forehead_img,mouth_img = [],[]
    mouth_intensity, forehead_intensity = 0,0


    if bboxs:
        center = bboxs[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
    
        eyes = eye_cascade.detectMultiScale(img, 1.03, 5, 0, (40, 40))
        x = bboxs[0]['bbox'][0]
        y = bboxs[0]['bbox'][1]
        w = bboxs[0]['bbox'][2]
        h = bboxs[0]['bbox'][3]
        cx = bboxs[0]['center'][0]
        cy = bboxs[0]['center'][1]


        # Se dibuja el rectangulo, arriba de los parametros de los ojos.
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

    cv2.imshow("Image", img)

    return img, forehead_img, mouth_img


while True:
    count += 1
    img, forehead_img, mouth_img = grabCam()
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
            print('spO2 Frente: ', Fsat)

        if(MAC_R > 0):
            Msat = spO2.SVR_PREDICT(MAC_R, MAC_G, MAC_B, MDC_R, MDC_G, MDC_B, regressor)
            print('spO2 Boca: ', Msat)

        count = 0

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
