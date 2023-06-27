import os
import cv2 as cv
import numpy as np
p = []
for i in os.listdir(r'C:\pythonproject\pythonProject1\twice'):
    p.append(i)

DIR = r'C:\pythonproject\pythonProject1\twice'
har = cv.CascadeClassifier('harr_face.xml')
features = []
labels = []
def create_train():
    for person in p:
        path = os.path.join(DIR,person)
        label = p.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect = har.detectMultiScale(gray,1.1,5)
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
#print('length of the features  is : ',len(features))
#print('length of labels = ',len(labels))
print('!---------training done ----------!')
features = np.array(features,dtype='object')
labels = np.array(labels)
face_recogniser = cv.face.LBPHFaceRecognizer_create()
#training the model on the features and labels list
face_recogniser.train(features,labels)
face_recogniser.save('faces_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)

