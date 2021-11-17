import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MASK_DETECTION_MODEL = load_model('data\models\model.h5')

face_detection = mp.solutions.face_detection.FaceDetection()

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

def detect_mask(frame):
    try:
        CATEGORIES = ['Mask', 'No Mask']

        img = frame.copy()

        x, y, w, h = get_detection(frame)
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, (250, 250))
        crop_img = np.expand_dims(crop_img, axis=0)
        prediction = MASK_DETECTION_MODEL.predict(crop_img)
        index = np.argmax(prediction)
        res = CATEGORIES[index]

        if index == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        frame = cv2.putText(frame, f"{res} {prediction[0]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return frame
    except:
        return frame
    
def make_square(image):
    height, width = image.shape[0:2]
    size = max(height, width)
    frame = np.zeros((size, size, 3), np.uint8)
    center_x, center_y = (size - width)//2, (size - height)//2
    frame[center_y:height+center_y, center_x:center_x+width] = image
    return frame

cap = cv2.VideoCapture(0)

while True:
    try: 
        ret, frame = cap.read()
        img = frame.copy()
    except:
        break
    try:
        CATEGORIES = ['Mask', 'No Mask']

        x, y, w, h = get_detection(frame)
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, (250, 250))
        crop_img = np.expand_dims(crop_img, axis=0)
        prediction = MASK_DETECTION_MODEL.predict(crop_img)
        index = np.argmax(prediction)
        res = CATEGORIES[index]

        if index == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        frame = cv2.putText(frame, f"{res} {prediction[0]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    except Exception as exception:
        print(exception)
    finally:
        frame = make_square(frame)
        frame = cv2.resize(frame, (750, 750), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Video Face Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()