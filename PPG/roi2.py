from cvzone.FaceDetectionModule import FaceDetector
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
detector = FaceDetector()
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('./mouth.xml')
forehead_img = 0 
mouth_img = 0

def grabCam():
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    
    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        print(bboxs)
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

        # cv2.rectangle(img, (exx,eyy), (exx+eww,eyy+ehh), (0,255,0), 2)

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

    print(forehead_intensity, mouth_intensity)
    cv2.imshow("Image", img)
    return forehead_intensity, mouth_intensity

while True:
    mouth_in, forehead_in = grabCam()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()