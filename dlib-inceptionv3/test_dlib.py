import dlib
import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
from json import JSONEncoder
from pygame import mixer

# Load the face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
face_landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open a connection to the camera
cap = cv2.VideoCapture(1)

model_folder = "models/"
model = load_model(f"{model_folder}inception3_model0414.h5")
mixer.init()
sound= mixer.Sound("../alarm.wav")

score = 0

def eye_preprocessing(eye):
    eye = cv2.resize(eye,(80,80))
    eye = eye/255
    eye = eye.reshape(80,80,3)
    eye = np.expand_dims(eye, axis=0)
    return eye

def get_eyes(landmarks):
    left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
    right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

    # Extract the eye region bounding boxes with a little extra margin
    left_eye_rect = (min(left_eye, key=lambda x: x[0])[0] - 30, min(left_eye, key=lambda x: x[1])[1] - 30,
                        max(left_eye, key=lambda x: x[0])[0] + 30, max(left_eye, key=lambda x: x[1])[1] + 30)
    right_eye_rect = (min(right_eye, key=lambda x: x[0])[0] - 30, min(right_eye, key=lambda x: x[1])[1] - 30,
                          max(right_eye, key=lambda x: x[0])[0] + 30, max(right_eye, key=lambda x: x[1])[1] + 30)
    
    return left_eye_rect, right_eye_rect

def draw_eye_rectangle(left_eye_rect, right_eye_rect):
    # Draw squares around the eye regions
    cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]),
                    (left_eye_rect[2], left_eye_rect[3]),
                    (0, 255, 0), 2)
    cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]),
                    (right_eye_rect[2], right_eye_rect[3]),
                    (0, 255, 0), 2)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    height,width = frame.shape[0:2]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    # Loop over the detected faces
    for face in faces:
        # Get the facial landmarks
        landmarks = face_landmark_predictor(gray, face)

        left_eye_rect, right_eye_rect = get_eyes(landmarks)

        draw_eye_rectangle(left_eye_rect, right_eye_rect)
        
        left_eye_region = frame[left_eye_rect[1]:left_eye_rect[3], left_eye_rect[0]:left_eye_rect[2]]
        right_eye_region = frame[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

        left_eye_frame = eye_preprocessing(left_eye_region)
        right_eye_frame = eye_preprocessing(right_eye_region)

        left_eye_prediction = model.predict(left_eye_frame)
        right_eye_prediction = model.predict(right_eye_frame)

        if left_eye_prediction[0][0] > 0.15 and right_eye_prediction[0][0] > 0.15:
            score+=1
            cv2.putText(frame,'closed',(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                    thickness=1,lineType=cv2.LINE_AA)
            if(score>10):
                try:
                    sound.play()
                except:
                    pass
        elif left_eye_prediction[0][1] > 0.92 and right_eye_prediction[0][1] > 0.92:
            score -= 1
            if(score<10):
                try:
                    sound.stop()
                except:
                    pass
            if (score<5):
                score=0
            cv2.putText(frame,'open',(200,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                    thickness=1,lineType=cv2.LINE_AA)
        
        cv2.putText(frame,'Score'+str(score),(50,height-30),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)

    # Display the frame with eye detections
    cv2.imshow("Eye Detection", frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
