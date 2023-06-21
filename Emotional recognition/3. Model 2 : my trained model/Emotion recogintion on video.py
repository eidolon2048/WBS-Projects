import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np


emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json file and create model
json_file = open('/Users/roma/Desktop/Emotional_recognition/streamlit_app/emotion_model4.json', 'r') # put your own file path here
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("/Users/roma/Desktop/Emotional_recognition/streamlit_app/emotion_model4.h5") # put your own file path here

#Camera set
frameWidth = 1280
frameHeight = 720

# frameWidth = 1920
# frameHeight = 1080

# frameWidth = 2048
# frameHeight = 1080

cap = cv2.VideoCapture(0) # or set 1
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)
cap.set(cv2.CAP_PROP_FPS, 30)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while cap.isOpened():
    success, img = cap.read()

    if success:
        ret,frame = cap.read()   #retiving
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Draw rectangle across the face
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(img=img, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 0, 200), 2, cv2.LINE_8)

        cv2.imshow("Your Emotion", frame)
        k = cv2.waitKey(100) & 0xFF 
        if k == 27: # press "esc" fpr stop video
            break

cap.release()
cv2.destroyAllWindows()