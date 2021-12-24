from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('PPG\mouth.xml')
fx = 0
fx2 = 0
while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    
    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        print(bboxs)
        center = bboxs[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
    
    eyes = eye_cascade.detectMultiScale(img, 1.03, 5, 0, (40, 40))
    bx = bboxs[0]['bbox'][0]
    by = bboxs[0]['bbox'][2]
    w = (bboxs[0]['center'][0] - bx)*2
    h = (bboxs[0]['center'][1] - by)*2 
    x = bx + w
    y = by + h

    # Se dibuja el rectangulo, arriba de los parametros de los ojos.
    for (ex,ey,ew,eh) in eyes:
        if abs(x-x+ex+ew) > w*0.51:
            fx2 = x+ex+ew
        else:
            fx = x+ex
        cv2.rectangle(img, (fx,y), (fx2,y+ey), (0,255,0), 2)
    
    ## Se aplica el reconociento de boca dentro del cuadro del rostro.
    mouth = mouth_cascade.detectMultiScale(img, 1.1, 20)
    ## Se elimina el reconocimiento de bocas sobre la mitad superior de la cara.
    for (mx,my,mw,mh) in mouth:
        if y+my > y + h/2:
            cv2.rectangle(img, (x+mx,y+my), (x+mx+mw,y+my+mh), (255,255,0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()