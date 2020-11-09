# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import face_recognition

imgFace=face_recognition.load_image_file('images/bill.jpg')
imgFace=cv2.cvtColor(imgFace,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('images/gates.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

facLoc=face_recognition.face_locations(imgFace)[0]
encodeFace= face_recognition.face_encodings(imgFace)[0]
cv2.rectangle(imgFace,(facLoc[3],facLoc[0]),(facLoc[1],facLoc[2]),(255,0,255),2)

facLocTest = face_recognition.face_locations(imgTest)[0]
encodeFaceTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facLocTest[3], facLocTest[0]), (facLocTest[1], facLocTest[2]), (255, 0, 255), 2)

results=face_recognition.compare_faces([encodeFace],encodeFaceTest)
faceDis = face_recognition.face_distance([encodeFace],encodeFaceTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('Bill',imgFace)
cv2.imshow('Gates', imgTest)
cv2.waitKey(0)
