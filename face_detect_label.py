import numpy as np
import os
import cv2 as cv

har = cv.CascadeClassifier('harr_face.xml')

p = []
for i in os.listdir(r'C:\pythonproject\pythonProject1\twice'):
    p.append(i)
#features = np.load('features.npy')
#labels = np.load('labels.npy')

face_recogniser = cv.face.LBPHFaceRecognizer_create()
face_recogniser.read('faces_trained.yml')

img = cv.imread("twice/sana/s1.jpg")
resized = cv.resize(img,(500,400),interpolation=cv.INTER_AREA)
cv.imshow("resized ",resized)

gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY)

#detect the face in the image
faces_rect = har.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    face_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recogniser.predict(face_roi)
    print('label = ',p[label],'with a confidence of ',confidence)

    cv.putText(resized,str(p[label]),(x,y),cv.FONT_HERSHEY_COMPLEX,1.2,(0,255,0),thickness= 2)
    cv.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
cv.imshow('Detected face',resized)


#for a video
capture =cv.VideoCapture(0)
while True:
    isTrue , frame=capture.read()
    resized2 = cv.resize(frame, (801, 472), interpolation=cv.INTER_AREA)
    grayv = cv.cvtColor(resized2,cv.COLOR_BGR2GRAY)
    faces_rect2= har.detectMultiScale(grayv, 1.5, 5)
    for (x, y, w, h) in faces_rect2:
        face_roi2 = grayv[y:y + h, x:x + w]

        label, confidence = face_recogniser.predict(face_roi2)
        print('label = ', p[label], 'with a confidence of ', confidence)

        cv.putText(resized2, str(p[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), thickness=2)

        cv.rectangle(resized2, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow('test2.0',resized2)
    if cv.waitKey(20) & 0xFF==ord('a'):
        break
capture.release()

cv.waitKey(0)
cv.destroyAllWindows()