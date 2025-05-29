
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
import random

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
age_model = load_model('model_age.h5')

def predict_age(face_image):
    face_image = cv2.resize(face_image, (224, 224))
    face_image = img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    face_image = preprocess_input(face_image)
    results = age_model.predict(face_image)
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages)
    return int(predicted_ages/random.uniform(1.2, 1.5))

cap = cv2.VideoCapture(0) 
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        age = predict_age(face_img)
        cv2.putText(img, "Age: " + str(age), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Age Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
