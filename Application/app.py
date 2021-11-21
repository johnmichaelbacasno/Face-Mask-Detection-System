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

# Load the model
MASK_DETECTION_MODEL = load_model('data\models\model.h5')

# Define mediapipe Face detector
face_detection = mp.solutions.face_detection.FaceDetection()

WHITE = "#FFFFFF"
BLUE = "#4C9FFC"

class VideoRunOutOfFrame(Exception):
    pass

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

def detect_face(frame):
    try:
        x, y, w, h = get_detection(frame)
        color = (0, 255, 0)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        frame = cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return frame
    except:
        return frame

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

class tkinterApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.title("Face Mask Detection System")
        self.geometry("1000x750")
        self.resizable(False, False)
        
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        #self.frames = {}
        self.show_frame(MenuPage)

    def show_frame(self, page, *args):
        """
        frame = page(self.container, self, *args)
        self.frames[page] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        self.frames[page].tkraise()
        """
        frame = page(self.container, self, *args)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()

class MenuPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(background="#23272a")
        self.controller = controller

        label_header = tk.Label(self, text="Choose\nInput Type", font=("Segoe UI Black", 50), bg="#23272a", fg="#ffffff")
        label_header.pack(pady=50)

        button_choice = tk.Button(self, text="Image", command=lambda: controller.show_frame(ImagePage), width=12, fg="#ffffff", bg="#7289da", bd=0, activebackground="#2c2f33", activeforeground="#ffffff", font=("Tw Cen MT", 20, "bold"), relief="flat")
        button_choice.pack(pady=20)
        
        button_voiced = tk.Button(self, text="Video", command=lambda: controller.show_frame(VideoPage), width=12, fg="#ffffff", bg="#7289da", bd=0, activebackground="#2c2f33", activeforeground="#ffffff", font=("Tw Cen MT", 20, "bold"), relief="flat")
        button_voiced.pack(pady=20)
        
        button_typing = tk.Button(self, text="Camera", command=lambda: controller.show_frame(CameraPage), width=12, fg="#ffffff", bg="#7289da", bd=0, activebackground="#2c2f33", activeforeground="#ffffff", font=("Tw Cen MT", 20, "bold"), relief="flat")
        button_typing.pack(pady=20)

        button_back = tk.Button(self, text="Back", command=self.controller.destroy, width=12, fg="#ffffff", bg="#f04747", bd=0, activebackground="#2c2f33", activeforeground="#ffffff", font=("Tw Cen MT", 20, "bold"), relief="flat")
        button_back.pack(pady=20)

class VideoPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(background="#23272a")
        
        self.video = None
        self.video_frame = None
        self.video_filename = None
        
        self.video_pause = True
        self.video_loop = False
        self.video_delay = 1
        self.video_end = False

        self.image_video_pause = ImageTk.PhotoImage(Image.open("assets/images/video_pause.png"))
        self.image_video_play = ImageTk.PhotoImage(Image.open("assets/images/video_play.png"))

        self.image_video_loop_off = ImageTk.PhotoImage(Image.open("assets/images/video_loop_off.png"))
        self.image_video_loop_on = ImageTk.PhotoImage(Image.open("assets/images/video_loop_on.png"))

        self.image_video_replay = ImageTk.PhotoImage(Image.open("assets/images/video_replay.png"))
        
        self.image_video_record_off = ImageTk.PhotoImage(Image.open("assets/images/video_record_off.png"))
        self.image_video_record_on = ImageTk.PhotoImage(Image.open("assets/images/video_record_on.png"))
        
        self.image_file_open = ImageTk.PhotoImage(Image.open("assets/images/file_open.png"))
        self.image_video_snapshot = ImageTk.PhotoImage(Image.open("assets/images/video_snapshot.png"))

        self.image_video_blank = ImageTk.PhotoImage(Image.open("assets/images/video_blank.png"))
        
        self.canvas = tk.Canvas(self, width=500, height=500)
        self.canvas.pack(anchor="center", padx=20, pady=20)
        self.canvas.create_image(0, 0, image=self.image_video_blank, anchor='nw')

        self.video_buttons = tk.Frame(self, background="#23272a")
        self.video_buttons.pack()

        self.button_pause = tk.Button(self.video_buttons, image=self.image_video_pause, command=self.switch_play, bd=0, background="#23272a", activebackground="#23272a")
        self.button_pause.grid(row=1, column=3, padx=15, pady=15)

        self.button_loop = tk.Button(self.video_buttons, image=self.image_video_loop_off, command=self.switch_loop, bd=0, background="#23272a", activebackground="#23272a")
        self.button_loop.grid(row=1, column=2, padx=15, pady=15)

        self.button_repeat = tk.Button(self.video_buttons, image=self.image_video_replay, command=self.replay_video, bd=0, background="#23272a", activebackground="#23272a")
        self.button_repeat.grid(row=1, column=1, padx=15, pady=15)

        self.button_record = tk.Button(self.video_buttons, image=self.image_video_record_off, command=self.video_record, bd=0, background="#23272a", activebackground="#23272a")
        self.button_record.grid(row=1, column=4, padx=15, pady=15)

        self.button_snapshot = tk.Button(self.video_buttons, image=self.image_video_snapshot, command=self.take_snapshot, bd=0, background="#23272a", activebackground="#23272a")
        self.button_snapshot.grid(row=1, column=5, padx=15, pady=15)

        self.button_open = tk.Button(self.video_buttons, image=self.image_file_open, command=self.open_file, bd=0, background="#23272a", activebackground="#23272a")
        self.button_open.grid(row=1, column=6, padx=15, pady=15)

        self.button_face_detect = tk.Button(self, text="Face Detect Off", width=50, command=self.face_detection_video)
        self.button_face_detect.pack(anchor="center")

        self.button_mask_detect = tk.Button(self, text="Mask Detect Off", width=50, command=self.mask_detection_video)
        self.button_mask_detect.pack(anchor="center")

        self.button_grey = tk.Button(self, text="Grey Off", width=50, command=self.grey_video)
        self.button_grey.pack(anchor="center")

        self.button_negative = tk.Button(self, text="Negative Off", width=50, command=self.negative_video)
        self.button_negative.pack(anchor="center")

        self.button_flip = tk.Button(self, text="Flip Off", width=50, command=self.flip_video)
        self.button_flip.pack(anchor="center")

        self.button_back = tk.Button(self, text="Back", width=50, command=self.destroy)
        self.button_back.pack(anchor="center")
    
    def take_snapshot(self):
        if self.video:
            cv2.imwrite(f"snapshots/image-{time.strftime('%Y-%m-%d-%H-%M-%S')}.jpg", cv2.cvtColor(self.video_frame, cv2.COLOR_RGB2BGR))
    
    def play_video(self):
        if self.video:
            if not self.video_pause:
                try:
                    self.video_frame = self.video.get_frame()
                    self.video_image = ImageTk.PhotoImage(image=Image.fromarray(self.video_frame))
                    self.canvas.create_image(0, 0, image=self.video_image, anchor='nw')
                except VideoRunOutOfFrame:
                    if self.video_loop:
                        self.replay_video()
                    else:
                        self.video_end = True
                        self.pause_video()
                    self.end_video_recording()
                self.after(self.video_delay, self.play_video)
            else:
                if self.video_end:
                    pass
                else:
                    #self.canvas.create_image(0, 0, image=self.image_video_blank, anchor='nw')
                    pass
    
    def switch_loop(self):
        if self.video:
            if self.video_loop:
                self.end_video_loop()
            else:
                self.start_video_loop()
    
    def start_video_loop(self):
        self.video_loop = True
        self.button_loop.config(image=self.image_video_loop_on)

    def end_video_loop(self):
        self.video_loop = False
        self.button_loop.config(image=self.image_video_loop_off)
    
    def switch_play(self):
        if self.video:
            if self.video_pause:
                self.resume_video()
            else:
                self.pause_video()
    
    def pause_video(self):
        self.video_pause = True
        self.button_pause.config(image=self.image_video_pause)
    
    def resume_video(self):
        if self.video_pause:
            self.video_pause = False
            self.button_pause.config(image=self.image_video_play)
            self.play_video()

    def replay_video(self):
        if self.video:
            self.video.refresh()
            self.video_end = False
            self.resume_video()

    def open_file(self):
        if self.video:
            self.pause_video()
            self.end_video_recording()
        
        self.video_filename = filedialog.askopenfilename(title="Open file", filetypes=(("MP4 files", "*.mp4"), ("WMV files", "*.wmv"), ("AVI files", "*.avi")))
        
        if self.video_filename:
            self.video = VideoCapture(self.video_filename)
            self.button_face_detect.config(text="Face Detect Off")
            self.button_grey.config(text="Grey Off")
            self.button_negative.config(text="Negative Off")
            self.button_flip.config(text="Flip Off")
            self.resume_video()
    
    def video_record(self):
        if self.video and not self.video_end:
            if self.video.recording:
                self.end_video_recording()
            else:
                self.start_video_recording()
    
    def start_video_recording(self):
        self.video.recording = True
        self.button_record.config(image=self.image_video_record_on)
        self.video.record_video()

    def end_video_recording(self):
        self.video.recording = False
        self.button_record.config(image=self.image_video_record_off)

    # Video Effects
    
    def face_detection_video(self):
        if self.video:
            if self.video.face_detection_is_enabled:
                self.end_face_detection()
            else:
                self.end_mask_detection()
                self.start_face_detection()
    
    def start_face_detection(self):
        self.video.face_detection_is_enabled = True
        self.button_face_detect.config(text="Face Detect On")
    
    def end_face_detection(self):
        self.video.face_detection_is_enabled = False
        self.button_face_detect.config(text="Face Detect Off")

    def mask_detection_video(self):
        if self.video:
            if self.video.mask_detection_is_enabled:
                self.end_mask_detection()
            else:
                self.end_face_detection()
                self.start_mask_detection()
    
    def start_mask_detection(self):
        self.video.mask_detection_is_enabled = True
        self.button_mask_detect.config(text="Mask Detect On")
    
    def end_mask_detection(self):
        self.video.mask_detection_is_enabled = False
        self.button_mask_detect.config(text="Mask Detect Off")
    
    def grey_video(self):
        if self.video:
            if self.video.grey_effect_is_enabled:
                self.video.grey_effect_is_enabled = False
                self.button_grey.config(text="Grey Off")
            else:
                self.video.grey_effect_is_enabled = True
                self.button_grey.config(text="Grey On")
    
    def negative_video(self):
        if self.video:
            if self.video.negative_effect_is_enabled:
                self.video.negative_effect_is_enabled = False
                self.button_negative.config(text="Negative Off")
            else:
                self.video.negative_effect_is_enabled = True
                self.button_negative.config(text="Negative On")
    
    def flip_video(self):
        if self.video:
            if self.video.flip_effect_is_enabled:
                self.video.flip_effect_is_enabled = False
                self.button_flip.config(text="Flip Off")
            else:
                self.video.flip_effect_is_enabled = True
                self.button_flip.config(text="Flip On")

class VideoCapture:
    def __init__(self, source=0):
        self.source = source
        self.vid = cv2.VideoCapture(self.source)
        
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", self.source)
        self.width = 500
        self.height = 500

        self.face_detection_is_enabled = False
        self.mask_detection_is_enabled = False
        self.negative_effect_is_enabled = False
        self.grey_effect_is_enabled = False
        self.flip_effect_is_enabled = False
        self.recording = False
        self.out = None
        self.record_frame = None

    def get_frame(self):
        if self.vid.isOpened():
            available, frame = self.vid.read()
            if available:
                if self.face_detection_is_enabled:
                    frame = detect_face(frame)
                if self.mask_detection_is_enabled:
                    frame = detect_mask(frame)
                if self.flip_effect_is_enabled:
                    frame = horizontal_flip(frame)
                if self.negative_effect_is_enabled:
                    frame = negative(frame)
                if self.grey_effect_is_enabled:
                    frame = grey(frame)

                frame = make_square(frame)
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                
                if self.recording:
                    self.record_frame = frame
                    #frame = cv2.putText(frame, "Recording...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise VideoRunOutOfFrame('No more video frames available.')
    
    def refresh(self):
        self.vid = cv2.VideoCapture(self.source)
    
    def record_video(self):
        if self.recording:
            self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.out = cv2.VideoWriter(f"recordings/video-{time.strftime('%Y-%m-%d-%H-%M-%S')}.mp4", self.fourcc, 20.0, (500, 500))
            self.video_recording_thread = Thread(target=self.record, args=[self.out,])
            self.video_recording_thread.start()
        else:
            self.out.release()
    
    def record(self, out):
        while self.recording:
            time.sleep(0.05)
            self.out.write(self.record_frame)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            if self.recording:
                self.out.release()

class ImagePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(background="#23272a")

        self.image = None
        self.delay = 1
        self.video_filename = None

        self.canvas = tk.Canvas(self, width=500, height=500)
        self.canvas.pack(anchor="center")
        self.photo = ImageTk.PhotoImage(Image.open("not_available.jpg"))
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.button_snapshot = tk.Button(self, text="Snapshot", width=50, command=self.take_snapshot)
        self.button_snapshot.pack(anchor="center")

        self.button_restore = tk.Button(self, text="Restore", width=50, command=self.restore_image)
        self.button_restore.pack(anchor="center")

        self.button_face_detect = tk.Button(self, text="Face Detect Off", width=50, command=self.face_detection_image)
        self.button_face_detect.pack(anchor="center")

        self.button_grey = tk.Button(self, text="Grey Off", width=50, command=self.grey_image)
        self.button_grey.pack(anchor="center")

        self.button_negative = tk.Button(self, text="Negative Off", width=50, command=self.negative_image)
        self.button_negative.pack(anchor="center")

        self.button_flip = tk.Button(self, text="Flip Off", width=50, command=self.flip_image)
        self.button_flip.pack(anchor="center")

        self.button_open_file = tk.Button(self, text="Open File", width=50, command=self.open_file)
        self.button_open_file.pack(anchor="center")

        self.button_back = tk.Button(self, text="Back", width=50, command=lambda: controller.show_frame(MenuPage))
        self.button_back.pack(anchor="center")
    
    def take_snapshot(self):
        if self.image:
            frame = self.image.get_frame()
            cv2.imwrite("snapshots/frame-" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def display_image(self):
        if self.image:
            frame = self.image.get_frame()
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
    
    def open_file(self):
        self.video_filename = filedialog.askopenfilename(title="Open file", filetypes=(("JPG files", "*.jpg"), ("PNG files", "*.png")))
        if self.video_filename:
            self.image = ImageCapture(self.video_filename)
            self.restore_image()

    def face_detection_image(self):
        if self.image:
            if self.image.face_detection_is_enabled:
                self.image.face_detection_is_enabled = False
                self.button_face_detect.config(text="Face Detect Off")
            else:
                self.image.face_detection_is_enabled = True
                self.button_face_detect.config(text="Face Detect On")
            self.display_image()
    
    def grey_image(self):
        if self.image:
            if self.image.grey_effect_is_enabled:
                self.image.grey_effect_is_enabled = False
                self.button_grey.config(text="Grey Off")
            else:
                self.image.grey_effect_is_enabled = True
                self.button_grey.config(text="Grey On")
            self.display_image()
    
    def negative_image(self):
        if self.image:
            if self.image.negative_effect_is_enabled:
                self.image.negative_effect_is_enabled = False
                self.button_negative.config(text="Negative Off")
            else:
                self.image.negative_effect_is_enabled = True
                self.button_negative.config(text="Negative On")
            self.display_image()
    
    def flip_image(self):
        if self.image:
            if self.image.flip_effect_is_enabled:
                self.image.flip_effect_is_enabled = False
                self.button_flip.config(text="Flip Off")
            else:
                self.image.flip_effect_is_enabled = True
                self.button_flip.config(text="Flip On")
            self.display_image()
    
    def restore_image(self):
        self.image.face_detection_is_enabled = False
        self.image.negative_effect_is_enabled = False
        self.image.grey_effect_is_enabled = False
        self.image.flip_effect_is_enabled = False

        self.button_face_detect.config(text="Face Detect Off")
        self.button_grey.config(text="Grey Off")
        self.button_negative.config(text="Negative Off")
        self.button_flip.config(text="Flip Off")

        self.display_image()

class ImageCapture:
    def __init__(self, source=None):
        self.source = source
        self.width = 500
        self.height = 500

        self.face_detection_is_enabled = False
        self.negative_effect_is_enabled = False
        self.grey_effect_is_enabled = False
        self.flip_effect_is_enabled = False
        
    def get_frame(self):
        frame = cv2.imread(self.source)
        if self.face_detection_is_enabled: frame = detect_face(frame)
        if self.negative_effect_is_enabled: frame = negative(frame)
        if self.flip_effect_is_enabled: frame = horizontal_flip(frame)
        frame = make_square(frame)
        if self.grey_effect_is_enabled: frame = grey(frame)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    app = tkinterApp()
    app.mainloop()