import cv2 as cv
cap = cv.VideoCapture(0) # '0' for webcam
while cap.isOpened():
    _, img = cap.read()
    cv.putText(img, 'hello', (30,300), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))   
    cv.imshow('MediaPipe Hands', img)
    if cv.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()