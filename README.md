## Introduction

This project aims to implement and optimize Haar Machine Learning model for Face recognition application
I would like to thank OpenCV library for helping me implement the model.

## Running scripts

Face Detection : _python face_detect.py [nameOfYourPictureFile]_

Live Cam: _python webcam_detect.py_

## Link face with personal data in sqlite Database

    python dataSetGeneratorCam.py (to enter your live face as well as the ID and name )
or  python dataSetGeneratorPics.py (enter face pic)
    python face-recognition-trainer.py (train the dataset of different faces)
    python face-recognition-detector.py (write name of the face detected )

## Documentation

If you want to understand how the code works, the details are here:

Face detection in static picture: https://realpython.com/blog/python/face-recognition-with-python/

Face detection in live camera: https://realpython.com/face-detection-in-python-using-a-webcam/
