import os
import cv2
import numpy as np
from PIL import Image
import sqlite3


from num2words import num2words
from subprocess import call

path = 'dataSet';
 
def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = [];
    IDs = [];
    for imagePath in imagePaths:
        if(imagePath!='dataSet\\Thumbs.db'): # To dump the hidden file Thumbs.db
            faceImg = Image.open(imagePath).convert('L');
            faceNp = np.array(faceImg, 'uint8');
            ID = int(os.path.split(imagePath)[-1].split('.')[1]);
            faces.append(faceNp);
            IDs.append(ID);
            cv2.imshow("training", faceNp);
            cv2.waitKey(10);
    return np.array(IDs), faces

def train():
    Ids, faces = getImagesWithID(path);
    rec.train(faces,Ids);
    rec.save('recognizer/trainingData.yml');

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

def dataCreate():
    cmd = 'Please_enter_your_ID_in_Py_Shell';
    call([cmd_beg+cmd+cmd_end], shell=True);
    id = raw_input('Enter your id: ');
    if(id=='0'):
        return;
    cmd = 'Please_enter_your_name';
    call([cmd_beg+cmd+cmd_end], shell=True);
    name = raw_input('Enter your Name: ');
    insorup(id,name);
    sampleNum=0;

    while(True):
        ret, img = cam.read();
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        faces = faceDetect.detectMultiScale(gray, 1.3, 5);
        for(x,y,w,h) in faces:
            sampleNum = sampleNum+1;
            cv2.imwrite("dataSet/User."+id+"."+str(sampleNum)+".jpg", gray[y:y+h,x:x+w]);
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2);
            cv2.waitKey(100);
        cv2.imshow("Face", img);
        cv2.waitKey(1);
        if(sampleNum>20):
            break;
    train();

def getProfile(id):
    conn=sqlite3.connect("FaceBase.db");
    cmd="SELECT * FROM People WHERE ID="+str(id);
    cursor=conn.execute(cmd);
    profile=None;
    for row in cursor:
        profile=row;
    conn.close();
    return profile;

cmd_beg= 'espeak '
cmd_end= ' 2>/dev/null' # To dump the std errors to /dev/null
 
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
cam.set(3,640);
rec = cv2.createLBPHFaceRecognizer();
train(); # don't leave dataset folder empty before running
rec.load('recognizer/trainingData.yml');
id = 0;
c=0
prename="";
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 5,1,0,4);
name="";
 
while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2);
        id, conf = rec.predict(gray[y:y+h, x:x+w]);

        if(conf<70):
            c=0;
            profile=getProfile(id);
            if(profile!=None):
                name=str(profile[1]);
                if(prename!=name):
                    cmd = 'Welcome_'+name;
                    call([cmd_beg+cmd+cmd_end], shell=True);
                prename=name;
                
        else:
            name="unknown"
            prename="";
            c=c+1;
            if(c>4):
                cmd = 'Welcome_'+name;
                call([cmd_beg+cmd+cmd_end], shell=True);
                dataCreate();
                c=0;
        cv2.cv.PutText(cv2.cv.fromarray(img),name, (x,y+h), font, 255);
    cv2.imshow("Face", img);
    if(cv2.waitKey(1)==ord('q')):
        cam.release();
        cv2.destroyAllWindows();
