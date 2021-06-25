import cv2 as cv
import os
cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier(haar_model)

cap = cv.VideoCapture(0) # '0' for webcam
while cap.isOpened():
    _, img = cap.read()
    frame_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]

    cv.putText(img, 'hello', (30,300), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))   
    cv.imshow('MediaPipe Hands', img)
    if cv.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()