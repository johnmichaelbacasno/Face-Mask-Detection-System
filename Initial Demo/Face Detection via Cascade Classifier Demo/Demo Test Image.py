import cv2
import numpy as np

classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
size = 5

def get_detection(frame):
    resized_down = cv2.resize(frame, (frame.shape[1] // size, frame.shape[0] // size))
    faces = classifier.detectMultiScale(resized_down)
    return faces

def make_square(image):
    height, width = image.shape[0:2]
    size = max(height, width)
    frame = np.zeros((size, size, 3), np.uint8)
    center_x, center_y = (size - width)//2, (size - height)//2
    frame[center_y:height+center_y, center_x:center_x+width] = image
    return frame

cap = cv2.imread('test.jpg')
img = cap
frame = img

try:
    for coor in get_detection(img):
        x, y, w, h = (value * size for value in coor)
        color = (0, 255, 0)
        frame = cv2.rectangle(img, (x, y) ,(x + w, y + h), color, 2)
        frame = cv2.putText(img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,cv2.LINE_AA)
except:
    pass
finally:
    frame = make_square(frame)
    frame = cv2.resize(frame, (750, 750), interpolation=cv2.INTER_LINEAR)

cv2.imshow('Image Face Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()