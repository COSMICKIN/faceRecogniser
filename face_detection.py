import cv2 as cv

img = cv.imread("ran.jpg")
#cv.imshow("face",img)
resized = cv.resize(img,(350,400),interpolation=cv.INTER_AREA)
cv.imshow("resized ",resized)
gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
#reading the harcascade xml
har = cv.CascadeClassifier('harr_face.xml')
faces_rect = har.detectMultiScale(gray,1.1,5)
print("the no of faces found =",len(faces_rect))
print(faces_rect)
for (x,y,w,h) in faces_rect:
    cv.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)

cv.imshow("detected face is  ",resized)
capture =cv.VideoCapture(0)
while True:
    isTrue , frame=capture.read()
    resized2 = cv.resize(frame, (801, 472), interpolation=cv.INTER_AREA)
    #cv.imshow("resized ", resized2)
    faces_rect2= har.detectMultiScale(resized2, 1.5, 5)
    for (x, y, w, h) in faces_rect2:
        cv.rectangle(resized2, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow('test',resized2)
    if cv.waitKey(20) & 0xFF==ord('a'):
        break
capture.release()
#cv.rectangle(gray,(227,106),(234+227,234+106),(0,255,0),2)
#cv.imshow("gray",gray)"""


cv.waitKey(0)
cv.destroyAllWindows()