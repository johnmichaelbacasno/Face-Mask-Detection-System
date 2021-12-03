import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import datetime, time
import os, sys
from threading import Thread
import tensorflow as tf
from tensorflow.keras.models import load_model

MASK_DETECTION_MODEL = load_model('data\models\model.h5')

face_detection = mp.solutions.face_detection.FaceDetection()

face_detected_count = 0
masked_detected_count = 0
umasked_detected_count = 0


class VideoRunOutOfFrame(Exception):
    pass

def make_square(image):
    height, width = image.shape[0:2]
    size = max(height, width)
    frame = np.zeros((size, size, 3), np.uint8)
    center_x, center_y = (size - width)//2, (size - height)//2
    frame[center_y:height+center_y, center_x:center_x+width] = image
    return frame

def grey(frame):
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def negative(frame):
    return cv2.bitwise_not(frame)

def vertical_flip(frame):
    return cv2.flip(frame, 0)

def horizontal_flip(frame):
    return cv2.flip(frame, 1)

'''
def get_detection(frame):
    height, width, channel = frame.shape
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(imgRGB)
    try:
        for count, detection in enumerate(result.detections):
            box = detection.location_data.relative_bounding_box
            x, y, w, h = int(box.xmin*width), int(box.ymin * height), int(box.width*width), int(box.height*height)
    except:
        pass
    return x, y, w, h

def detect_face(frame):
    try:
        x, y, w, h = get_detection(frame)
        color = (0, 255, 0)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        frame = cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    except:
        pass
    return frame

def detect_mask(frame):
    try:
        CATEGORIES = ['Mask', 'No Mask']
        image_copy = frame.copy()

        x, y, w, h = get_detection(frame)
        cropped_image = image_copy[y:y+h, x:x+w]
        cropped_image = cv2.resize(cropped_image, (250, 250))
        cropped_image = np.expand_dims(cropped_image, axis=0)
        prediction = MASK_DETECTION_MODEL.predict(cropped_image)
        index = np.argmax(prediction)
        response = CATEGORIES[index]
        color =  (0, 255, 0) if index == 0 else (0, 0, 255)
        
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        frame = cv2.putText(frame, f"{response} {format(prediction[0][0]*100, '.2f')}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
    except:
        pass
    
    return frame
'''

classifier = cv2.CascadeClassifier('data/models/haarcascade_frontalface_default.xml')
size = 7

def get_detection(frame):
    '''
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)
    '''
    resized_down = cv2.resize(frame, (frame.shape[1] // size, frame.shape[0] // size))
    faces = classifier.detectMultiScale(resized_down)
    return faces

def detect_face(frame):
    global size, face_detected_count
    try:
        image_copy = frame.copy()
        faces_count = 0
        for coor in get_detection(image_copy):
            x, y, w, h = (value * size for value in coor)
            color = (0, 255, 0)
            frame = cv2.rectangle(image_copy, (x, y) ,(x + w, y + h), color, 3)
            frame = cv2.putText(image_copy, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,cv2.LINE_AA)
            faces_count += 1
        face_detected_count = faces_count
    except Exception as e:
        print(e)
    
    return frame

def detect_mask(frame):
    global size, face_detected_count, masked_detected_count, umasked_detected_count
    try:
        CATEGORIES = ['Mask', 'No Mask']
        image_copy = frame.copy()
        
        faces_count = 0
        mask_count = 0
        unmasked_count = 0
        
        for coor in get_detection(image_copy):
            x, y, w, h = (value * size for value in coor)
            cropped_image = image_copy[y:y+h, x:x+w]
            cropped_image = cv2.resize(cropped_image, (250, 250))
            cropped_image = np.expand_dims(cropped_image, axis=0)
            prediction = MASK_DETECTION_MODEL.predict(cropped_image)
            index = np.argmax(prediction)
            response = CATEGORIES[index]

            if index == 0:
                color = (0, 255, 0)
                mask_count += 1
            else:
                color = (0, 0, 255)
                unmasked_count += 1
            
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            frame = cv2.putText(frame, f"{response} {format(prediction[0][0]*100, '.2f')}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            faces_count += 1
        
        face_detected_count = faces_count
        masked_detected_count = mask_count
        umasked_detected_count = unmasked_count
    
    except:
        pass
    
    return frame