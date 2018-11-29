# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:35:26 2018

@author: Fakhrovski
"""

import cv2
#import logging as log
#import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# log.basicConfig(filename='webcamLog.log',level=log.INFO)

video_snap = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_snap.isOpened():
        print('Error in loading camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_snap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        #log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)


# When everything is done, release the capture
video_snap.release()
cv2.destroyAllWindows()
