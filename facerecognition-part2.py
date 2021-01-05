# Import the modules
import cv2
import numpy as np
from os import listdir
import pickle
from os.path import isfile, join
import time

from numpy.core.records import array

models = {
    "ananya": {
        "data_path": "face/ananya/",
        "files": [],
        "model": None
    },
    "komal": {
        "data_path": "face/komal/",
        "files": [],
        "model": None
    },    
    "Arunima": {
        "data_path": "face/Arunima/",
        "files": [],
        "model": None
    },    
    "Ibrahim": {
        "data_path": "face/Ibrahim/",
        "files": [],
        "model": None
    }
}
for key in models:
    print("Started training model for " + key)
    models[key]["files"] = [f for f in listdir(models[key]["data_path"]) if isfile(join(models[key]["data_path"], f))]
    Training_Data, Labels = [], []

    for i, files in enumerate(models[key]["files"]):
        image_path = models[key]["data_path"] + models[key]["files"][i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append( np.asarray( images, dtype=np.uint8))
        Labels.append(i)
    
    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)

    # Initialize facial recognizer
    models[key]["model"] =  cv2.face.LBPHFaceRecognizer_create()
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()

    #Training_Data), np.asarray(Labels))

    # Let's train our model
    models[key]["model"].train(np.asarray( Training_Data), np.asarray(Labels))
    print("Model trained sucessefully for " + key)



# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')

# Function to detect face
def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale( gray, 1.3, 5)
    # If face not found return blank region
    if faces is ():
        return [img, [], None]
    # Obtain Region of face
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    
    return [img, roi, faces[0]]


# Open Webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    ar = face_detector(frame)
    image = ar[0] 
    face=ar[1] 
    pos=ar[2]


    time.sleep(0.06)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # print(face)
        foundFace = False
        user = None
        confidence = 0
        for key in models:
            if foundFace == True:
                break
            results = models[key]["model"].predict(face)
            if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/400) )
                if confidence > 88:
                    user = key
                    foundFace = True
                # display_string = str(confidence) + '% Confident it is User'
        
        posX = pos[0] +10
        posY = pos[0] - 10
        cv2.putText(image, "Accuracy " + str(confidence) + "%", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        if foundFace == True:
            cv2.putText(image, user, (posX, posY), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        else:
            cv2.putText(image, "Unknown ", (posX, posY), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        
        cv2.imshow('Face Recognition', image )
    # Raise exception in case, no image is found
    except Exception as e:
        # cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        # cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Accuracy 0%", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        cv2.imshow('Face Recognition', image )
        
        pass
    # Breaks loop when enter is pressed
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

# Release and destroyAllWindows
cap.release()
cv2.destroyAllWindows()