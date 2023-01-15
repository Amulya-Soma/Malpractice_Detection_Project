import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.preprocessing import image
import cv2

import datetime
import time

mymodel = load_model('malpractice.model')
labels = ['No malpractice', 'Malpractise']
cap = cv2.VideoCapture(0)
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while cap.isOpened():
    _, img = cap.read()
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in face:
        face_img = img[y:y + h, x:x + w]
        cv2.imwrite('temp2.jpg', img)
        test_image = image.load_img('temp2.jpg', target_size=(256, 256, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)


        pred = mymodel.predict(test_image)
        thres = (pred>0.5)*1
        print(thres)
        ind = np.argmax(thres)
        y = (labels[ind])
        print(y)


        cv2.putText(img, y, (100,100), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,255))
        
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()