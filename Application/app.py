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

face_detection = mp.solutions.face_detection.FaceDetection()

WHITE = "#FFFFFF"
BLUE = "#4C9FFC"

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
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def negative(frame):
    return cv2.bitwise_not(frame)

def flip(frame):
    return cv2.flip(frame, 1)

def detect_face(frame):
    img = frame.copy()
    try:
        x, y, w, h = get_detection(frame)
        color = (0, 255, 0)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        frame = cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return frame
    except Exception as error:
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

        self.vid = None
        self.pause = True
        self.delay = 1
        self.filename = None
        self.video_end = False

        #IMAGES
        self.pause_img = ImageTk.PhotoImage(Image.open("assets/images/off.png"))
        self.play_img = ImageTk.PhotoImage(Image.open("assets/images/on.png"))

        self.canvas = tk.Canvas(self, width=500, height=500)
        self.canvas.pack(anchor="center")
        self.photo = ImageTk.PhotoImage(Image.open("not_available.jpg"))
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.btn_open_file = tk.Button(self, text="Open File", width=50, command=self.open_file)
        self.btn_open_file.pack(anchor="center")

        self.btn_snapshot = tk.Button(self, text="Snapshot", width=50, command=self.take_snapshot)
        self.btn_snapshot.pack(anchor="center")
        
        self.btn_pause = tk.Button(self, image=self.pause_img, command=self.switch_play, bd=0, background="#23272a", activebackground="#23272a")
        self.btn_pause.pack(anchor="center")

        self.btn_replay = tk.Button(self, text="Replay", width=50, command=self.replay_video)
        self.btn_replay.pack(anchor="center")

        self.btn_record = tk.Button(self, text="Record Off", width=50, command=self.video_record)
        self.btn_record.pack(anchor="center")

        self.btn_face_detect = tk.Button(self, text="Face Detect Off", width=50, command=self.face_detection_video)
        self.btn_face_detect.pack(anchor="center")

        self.btn_grey = tk.Button(self, text="Grey Off", width=50, command=self.grey_video)
        self.btn_grey.pack(anchor="center")

        self.btn_negative = tk.Button(self, text="Negative Off", width=50, command=self.negative_video)
        self.btn_negative.pack(anchor="center")

        self.btn_flip = tk.Button(self, text="Flip Off", width=50, command=self.flip_video)
        self.btn_flip.pack(anchor="center")

        self.btn_back = tk.Button(self, text="Back", width=50, command=self.destroy)
        self.btn_back.pack(anchor="center")

    def take_snapshot(self):
        if self.vid and not self.video_end:
            frame = self.vid.get_frame()
            cv2.imwrite(f"snapshots/image-{time.strftime('%Y-%m-%d-%H-%M-%S')}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def play_video(self):
        if self.vid:
            try:
                frame = self.vid.get_frame()
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
            except:
                # When video ends:
                # Pause Video
                self.pause_video()
                # Stop Recording
                self.video_end = True
                self.vid.recording = False
                self.btn_record.config(text="Record Off")
            if not self.pause:
                self.after(self.delay, self.play_video)
    
    def pause_video(self):
        self.pause = True
    
    def switch_play(self):
        if self.vid:
            if self.pause:
                self.resume_video()
                self.btn_pause.config(image=self.pause_img)
            else:
                self.pause_video()
                self.btn_pause.config(image=self.play_img)
    
    def resume_video(self):
        if self.pause:
            self.pause = False
            self.play_video()
    
    def replay_video(self):
        if self.vid:
            self.vid.refresh()
            self.video_end = False
            self.btn_pause.config(text="Pause")
            self.resume_video()
    
    def open_file(self):
        self.pause_video()
        self.filename = filedialog.askopenfilename(title="Open file", filetypes=(("MP4 files", "*.mp4"), ("WMV files", "*.wmv"), ("AVI files", "*.avi")))
        if self.filename:
            self.vid = VideoCapture(self.filename)
            self.btn_face_detect.config(text="Face Detect Off")
            self.btn_grey.config(text="Grey Off")
            self.btn_negative.config(text="Negative Off")
            self.btn_flip.config(text="Flip Off")
            self.resume_video()
    
    def video_record(self):
        if self.vid and not self.video_end:
            if self.vid.recording:
                self.vid.recording = False
                self.btn_record.config(text="Record Off")
            else:
                self.vid.recording = True
                self.btn_record.config(text="Record On")
                self.vid.record_video()

    def face_detection_video(self):
        if self.vid:
            if self.vid.face_detection_is_enabled:
                self.vid.face_detection_is_enabled = False
                self.btn_face_detect.config(text="Face Detect Off")
            else:
                self.vid.face_detection_is_enabled = True
                self.btn_face_detect.config(text="Face Detect On")
    
    def grey_video(self):
        if self.vid:
            if self.vid.grey_effect_is_enabled:
                self.vid.grey_effect_is_enabled = False
                self.btn_grey.config(text="Grey Off")
            else:
                self.vid.grey_effect_is_enabled = True
                self.btn_grey.config(text="Grey On")
    
    def negative_video(self):
        if self.vid:
            if self.vid.negative_effect_is_enabled:
                self.vid.negative_effect_is_enabled = False
                self.btn_negative.config(text="Negative Off")
            else:
                self.vid.negative_effect_is_enabled = True
                self.btn_negative.config(text="Negative On")
    
    def flip_video(self):
        if self.vid:
            if self.vid.flip_effect_is_enabled:
                self.vid.flip_effect_is_enabled = False
                self.btn_flip.config(text="Flip Off")
            else:
                self.vid.flip_effect_is_enabled = True
                self.btn_flip.config(text="Flip On")

class VideoCapture:
    def __init__(self, source=0):
        self.source = source
        self.vid = cv2.VideoCapture(self.source)
        
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", self.source)
        self.width = 500
        self.height = 500

        #self.switch_detect_mask = False
        self.face_detection_is_enabled = False
        self.negative_effect_is_enabled = False
        self.grey_effect_is_enabled = False
        self.flip_effect_is_enabled = False
        self.recording = False
        self.out = None
        self.record_frame = None

    def get_frame(self):
        if self.vid.isOpened():
            _, frame = self.vid.read()
            if self.face_detection_is_enabled: frame = detect_face(frame)
            if self.negative_effect_is_enabled:frame = negative(frame)
            if self.flip_effect_is_enabled: frame = flip(frame)
            frame = make_square(frame)
            if self.grey_effect_is_enabled: frame = grey(frame)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if self.recording:
                self.record_frame = frame
                frame = cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def refresh(self):
        self.vid = cv2.VideoCapture(self.source)
    
    def record_video(self):
        if self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.out = cv2.VideoWriter(f"recordings/video-{time.strftime('%Y-%m-%d-%H-%M-%S')}.mp4", fourcc, 20.0, (500, 500))
            thread = Thread(target=self.record, args=[self.out,])
            thread.start()
        else:
            self.out.release()
    
    def record(self, out):
        while self.recording:
            time.sleep(0.05)
            self.out.write(self.record_frame)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class ImagePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(background="#23272a")

        self.image = None
        self.delay = 1
        self.filename = None

        self.canvas = tk.Canvas(self, width=500, height=500)
        self.canvas.pack(anchor="center")
        self.photo = ImageTk.PhotoImage(Image.open("not_available.jpg"))
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.btn_snapshot = tk.Button(self, text="Snapshot", width=50, command=self.take_snapshot)
        self.btn_snapshot.pack(anchor="center")

        self.btn_restore = tk.Button(self, text="Restore", width=50, command=self.restore_image)
        self.btn_restore.pack(anchor="center")

        self.btn_face_detect = tk.Button(self, text="Face Detect Off", width=50, command=self.face_detection_image)
        self.btn_face_detect.pack(anchor="center")

        self.btn_grey = tk.Button(self, text="Grey Off", width=50, command=self.grey_image)
        self.btn_grey.pack(anchor="center")

        self.btn_negative = tk.Button(self, text="Negative Off", width=50, command=self.negative_image)
        self.btn_negative.pack(anchor="center")

        self.btn_flip = tk.Button(self, text="Flip Off", width=50, command=self.flip_image)
        self.btn_flip.pack(anchor="center")

        self.btn_open_file = tk.Button(self, text="Open File", width=50, command=self.open_file)
        self.btn_open_file.pack(anchor="center")

        self.btn_back = tk.Button(self, text="Back", width=50, command=lambda: controller.show_frame(MenuPage))
        self.btn_back.pack(anchor="center")

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
        self.filename = filedialog.askopenfilename(title="Open file", filetypes=(("JPG files", "*.jpg"), ("PNG files", "*.png")))
        if self.filename:
            self.image = ImageCapture(self.filename)
            self.restore_image()

    def face_detection_image(self):
        if self.image:
            if self.image.face_detection_is_enabled:
                self.image.face_detection_is_enabled = False
                self.btn_face_detect.config(text="Face Detect Off")
            else:
                self.image.face_detection_is_enabled = True
                self.btn_face_detect.config(text="Face Detect On")
            self.display_image()
    
    def grey_image(self):
        if self.image:
            if self.image.grey_effect_is_enabled:
                self.image.grey_effect_is_enabled = False
                self.btn_grey.config(text="Grey Off")
            else:
                self.image.grey_effect_is_enabled = True
                self.btn_grey.config(text="Grey On")
            self.display_image()
    
    def negative_image(self):
        if self.image:
            if self.image.negative_effect_is_enabled:
                self.image.negative_effect_is_enabled = False
                self.btn_negative.config(text="Negative Off")
            else:
                self.image.negative_effect_is_enabled = True
                self.btn_negative.config(text="Negative On")
            self.display_image()
    
    def flip_image(self):
        if self.image:
            if self.image.flip_effect_is_enabled:
                self.image.flip_effect_is_enabled = False
                self.btn_flip.config(text="Flip Off")
            else:
                self.image.flip_effect_is_enabled = True
                self.btn_flip.config(text="Flip On")
            self.display_image()
    
    def restore_image(self):
        self.image.face_detection_is_enabled = False
        self.image.negative_effect_is_enabled = False
        self.image.grey_effect_is_enabled = False
        self.image.flip_effect_is_enabled = False

        self.btn_face_detect.config(text="Face Detect Off")
        self.btn_grey.config(text="Grey Off")
        self.btn_negative.config(text="Negative Off")
        self.btn_flip.config(text="Flip Off")

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
        if self.flip_effect_is_enabled: frame = flip(frame)
        frame = make_square(frame)
        if self.grey_effect_is_enabled: frame = grey(frame)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

app = tkinterApp()
app.mainloop()