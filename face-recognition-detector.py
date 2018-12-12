import numpy as np
import os
import cv2
from PIL import Image
import pickle, sqlite3

face_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/training_data.yml")

#To train using images captured or saved online
# img = cv2.imread("him.jpg")

def getProfile(Id):
    conn=sqlite3.connect("FaceBase.db")
    query="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

#to train using frames from video
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    #comment the next line and make sure the image being read is names img when using imread
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Hiding the eye detector for now
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        nbr_predicted, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 70:
            profile=getProfile(nbr_predicted)
            if profile != None:
                cv2.putText(img, "Prenom: "+str(profile[1]), (x, y+h+30), font, 0.4, (0, 0, 255), 1);
                cv2.putText(img, "Nom: " + str(profile[2]), (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
        else:
            cv2.putText(img, "Prenom: Unknown", (x, y + h + 30), font, 0.4, (0, 0, 255), 1);
            cv2.putText(img, "Nom: Unknown", (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
 

    cv2.imshow('img', img)
    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()

