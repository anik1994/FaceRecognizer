import os
import cv2
import numpy as np
from PIL import Image
import sqlite3
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def insorup(ID, Name):
    conn=sqlite3.connect("FaceBase.db");
    cmd="SELECT ID FROM People WHERE ID="+str(ID);
    cursor=conn.execute(cmd);
    isRecordExist=0;
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE People SET Name='"+str(Name)+"' WHERE ID="+str(ID);
    else:
        cmd="INSERT INTO People(ID,Name) Values("+str(ID)+",'"+str(Name)+"')";
    conn.execute(cmd);
    conn.commit();
    conn.close();

Id=raw_input('enter your id')
#name = raw_input('Enter your Name: ');
#insorup(id,name);
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
