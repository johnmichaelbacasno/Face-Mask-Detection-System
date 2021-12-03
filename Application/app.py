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

class tkinterApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.title("Face Mask Detection System")
        self.geometry("1000x725")
        self.resizable(False, False)
        
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        '''
        self.frames = {}
        '''
        self.show_frame(MenuPage)

    def show_frame(self, page, *args):
        '''
        frame = page(self.container, self, *args)
        self.frames[page] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        self.frames[page].tkraise()
        '''
        frame = page(self.container, self, *args)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()

class Warning:
    def __init__(self, root, title, message):
        self.root = root
        self.root.title(title)
        self.root.geometry("450x325")
        self.root.resizable(width=False, height=False)
        self.root.configure(background="#000C18")

        frame_main = tk.Frame(self.root, background="#000C18")
        frame_main.pack(pady=20)

        label_warning_icon = tk.Label(frame_main, text="⚠", font=("Tw Cen MT", 50, "bold"), bg="#000C18", fg="#E62A32")
        label_warning_icon.pack(padx=20, pady=10)

        label_message = tk.Label(frame_main, text=message, font=("Tw Cen MT", 20), bg="#000C18", fg="#FFFFFF")
        label_message.pack(padx=20, pady=15)

        frame_button = tk.Frame(self.root, background="#000C18")
        frame_button.pack(pady=20)

        button_quit = tk.Button(frame_button, text="OK", command=self.root.destroy, height=1, width=8, fg="#FFFFFF", bg="#E62A32", bd=0, activebackground="#2c2f33", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="flat")
        button_quit.pack()

class AskQuit:
    def __init__(self, root, controller, title):
        self.root = root
        self.root.title(title)
        self.root.geometry("450x325")
        self.root.resizable(width=False, height=False)
        self.root.configure(background="#000C18")
        self.controller = controller

        frame_main = tk.Frame(self.root, background="#000C18")
        frame_main.pack(pady=20)

        label_warning_icon = tk.Label(frame_main, text="✋", font=("UD Digi Kyokasho NK-B", 50), bg="#000C18", fg="#E62A32")
        label_warning_icon.pack(padx=20, pady=10)

        label_message = tk.Label(frame_main, text="Are you sure you want to quit?", font=("Tw Cen MT", 20), bg="#000C18", fg="#FFFFFF")
        label_message.pack(padx=20, pady=15)

        frame_button = tk.Frame(self.root, background="#000C18")
        frame_button.pack(pady=20)

        button_yes = tk.Button(frame_button, text="Yes", command=self.yes, height=1, width=8, fg="#FFFFFF", bg="#E62A32", bd=0, activebackground="#2c2f33", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="flat")
        button_yes.grid(row=0, column=0, padx=20)

        button_no = tk.Button(frame_button, text="No", command=self.no, height=1, width=8, fg="#FFFFFF", bg="#00AAEB", bd=0, activebackground="#2c2f33", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="flat")
        button_no.grid(row=0, column=1, padx=20)

    def yes(self):
        self.controller.destroy()
    
    def no(self):
        self.root.destroy()

class MenuPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(background="#000C18")
        self.controller = controller

        label_header = tk.Label(self, text="Choose\nInput Type", font=("Tw Cen MT", 60, "bold"), bg="#000C18", fg="#FFFFFF")
        label_header.pack(pady=50)

        button_image = tk.Button(self, text="Image", command=lambda: controller.show_frame(ImagePage), width=12, fg="#FFFFFF", bg="#00AAEB", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="raised")
        button_image.pack(pady=20)
        
        button_video = tk.Button(self, text="Video", command=lambda: controller.show_frame(VideoPage), width=12, fg="#FFFFFF", bg="#00AAEB", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="raised")
        button_video.pack(pady=20)
        
        button_camera = tk.Button(self, text="Camera", command=self.browse_camera_page, width=12, fg="#FFFFFF", bg="#00AAEB", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="raised")
        button_camera.pack(pady=20)

        button_quit = tk.Button(self, text="Quit", command=lambda: AskQuit(tk.Toplevel(self), self.controller, "Quit"), width=12, fg="#FFFFFF", bg="#E62A32", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 20, "bold"), relief="raised")
        button_quit.pack(pady=20)

    def browse_camera_page(self):
        try:
            self.controller.show_frame(CameraPage)
        except ValueError:
            Warning(tk.Toplevel(self), "Warning", "No webcam available")
        except:
            Warning(tk.Toplevel(self), "Warning", "An error occured")


class VideoPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(background="#000C18")
        
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
        
        self.canvas = tk.Canvas(self, width=498, height=498)
        self.canvas.pack(anchor="center", padx=20, pady=20)
        self.canvas.create_image(0, 0, image=self.image_video_blank, anchor='nw')
        
        '''
        self.canvas = tk.Label(self, width=500, height=500)
        self.canvas.pack(anchor="center", padx=20, pady=20)
        self.canvas.configure(image=self.image_video_blank, anchor='nw')
        '''

        self.video_buttons = tk.Frame(self, background="#000C18")
        self.video_buttons.pack()

        self.button_pause = tk.Button(self.video_buttons, image=self.image_video_pause, command=self.switch_play, bd=0, background="#000C18", activebackground="#000C18")
        self.button_pause.grid(row=1, column=3, padx=15, pady=15)

        self.button_loop = tk.Button(self.video_buttons, image=self.image_video_loop_off, command=self.switch_loop, bd=0, background="#000C18", activebackground="#000C18")
        self.button_loop.grid(row=1, column=2, padx=15, pady=15)

        self.button_repeat = tk.Button(self.video_buttons, image=self.image_video_replay, command=self.replay_video, bd=0, background="#000C18", activebackground="#000C18")
        self.button_repeat.grid(row=1, column=1, padx=15, pady=15)

        self.button_record = tk.Button(self.video_buttons, image=self.image_video_record_off, command=self.video_record, bd=0, background="#000C18", activebackground="#000C18")
        self.button_record.grid(row=1, column=4, padx=15, pady=15)

        self.button_snapshot = tk.Button(self.video_buttons, image=self.image_video_snapshot, command=self.take_snapshot, bd=0, background="#000C18", activebackground="#000C18")
        self.button_snapshot.grid(row=1, column=5, padx=15, pady=15)

        self.button_open = tk.Button(self.video_buttons, image=self.image_file_open, command=self.open_file, bd=0, background="#000C18", activebackground="#000C18")
        self.button_open.grid(row=1, column=6, padx=15, pady=15)

        self.video_button_effects = tk.Frame(self, background="#000C18")
        self.video_button_effects.pack()

        self.button_face_detect = tk.Button(self.video_button_effects, text="Face Detect", command=self.face_detection_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_face_detect.grid(row=1, column=0, padx=15, pady=15)

        self.button_mask_detect = tk.Button(self.video_button_effects, text="Mask Detect", command=self.mask_detection_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_mask_detect.grid(row=1, column=1, padx=15, pady=15)

        self.button_grey = tk.Button(self.video_button_effects, text="Grey", command=self.grey_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_grey.grid(row=1, column=2, padx=15, pady=15)

        self.button_negative = tk.Button(self.video_button_effects, text="Negative", command=self.negative_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_negative.grid(row=1, column=3, padx=15, pady=15)

        self.button_horizontal_flip = tk.Button(self.video_button_effects, text="H-Flip", command=self.horizontal_flip_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_horizontal_flip.grid(row=1, column=4, padx=15, pady=15)

        self.button_vertical_flip = tk.Button(self.video_button_effects, text="V-Flip", command=self.vertical_flip_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_vertical_flip.grid(row=1, column=5, padx=15, pady=15)

        self.button_back = tk.Button(self, text="Back", command=self.end_page, width=12, fg="#FFFFFF", bg="#E62A32", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12, "bold"), relief="raised")
        self.button_back.place(bordermode="outside", x=35, y=20)

        self.info = tk.Frame(self, background="#000C18")
        self.info.place(bordermode="outside", x=800, y=15)

        self.label_face_detected = tk.Label(self.info, text="Face Detected",  font=("Tw Cen MT Condensed", 12), bg="#000C18", fg="#FFFFFF")
        self.label_face_detected.grid(row=1, column=0, padx=5, pady=5, sticky="W")

        self.count_face_detected = tk.Label(self.info, text=0,  font=("Tw Cen MT Condensed", 12, "bold"), bg="#000C18", fg="#FFFFFF")
        self.count_face_detected.grid(row=1, column=1, padx=5, pady=5, sticky="W")
        
        self.label_masked_detected = tk.Label(self.info, text="Masked Detected", font=("Tw Cen MT Condensed", 12), bg="#000C18", fg="#FFFFFF")
        self.label_masked_detected.grid(row=2, column=0, padx=5, pady=5, sticky="W")

        self.count_masked_detected = tk.Label(self.info, text=0, font=("Tw Cen MT Condensed", 12, "bold"), bg="#000C18", fg="#FFFFFF")
        self.count_masked_detected.grid(row=2, column=1, padx=5, pady=5, sticky="W")

        self.label_unmasked_detected = tk.Label(self.info, text="Unmasked Detected", font=("Tw Cen MT Condensed", 12), bg="#000C18", fg="#FFFFFF")
        self.label_unmasked_detected.grid(row=3, column=0, padx=5, pady=5, sticky="W")

        self.count_unmasked_detected = tk.Label(self.info, text=0, font=("Tw Cen MT Condensed", 12, "bold"), bg="#000C18", fg="#FFFFFF")
        self.count_unmasked_detected.grid(row=3, column=1, padx=5, pady=5, sticky="W")

        self.info_time = tk.Frame(self, background="#000C18")
        self.info_time.place(bordermode="outside", x=800, y=450)
        
        self.label_date = tk.Label(self.info_time, text=time.strftime('%B %d, %Y'),  font=("Tw Cen MT Condensed", 15), bg="#000C18", fg="#FFFFFF")
        self.label_date.grid(row=0, column=0, padx=5, pady=3, sticky="W")

        self.label_time = tk.Label(self.info_time, text=time.strftime('%H:%M %p'),  font=("Tw Cen MT Condensed", 13), bg="#000C18", fg="#FFFFFF")
        self.label_time.grid(row=1, column=0, padx=5, pady=3, sticky="W")

        self.update_time()
    
    def update_time(self):
        self.label_date.configure(text=time.strftime('%B %d, %Y'))
        self.label_time.configure(text=time.strftime('%H:%M:%S %p'))
        self.label_time.after(1000, self.update_time)
    
    def update_info(self):
        self.count_face_detected.configure(text=face_detected_count)
        self.count_masked_detected.configure(text=masked_detected_count)
        self.count_unmasked_detected.configure(text=umasked_detected_count)
    
    def reset_count(self):
        global face_detected_count, masked_detected_count, umasked_detected_count
        face_detected_count = 0
        masked_detected_count = 0
        umasked_detected_count = 0
    
    def end_page(self):
        self.reset_count()
        self.destroy()
    
    def take_snapshot(self):
        if self.video:
            try:
                cv2.imwrite(f"snapshots/image-{time.strftime('%Y-%m-%d-%H-%M-%S')}.jpg", cv2.cvtColor(self.video_frame, cv2.COLOR_RGB2BGR))
            except:
                pass
    
    def play_video(self):
        if self.video:
            self.update_info()
            if not self.video_pause:
                try:
                    
                    self.video_frame = self.video.get_frame()
                    self.video_image = ImageTk.PhotoImage(image=Image.fromarray(self.video_frame))
                    self.canvas.create_image(0, 0, image=self.video_image, anchor='nw')
                    
                    '''
                    self.video_frame = self.video.get_frame()
                    self.video_image = ImageTk.PhotoImage(image=Image.fromarray(self.video_frame))
                    self.canvas.configure(image=self.video_image, anchor='nw')
                    '''

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
                    # self.canvas.create_image(0, 0, image=self.image_video_blank, anchor='nw')
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
            self.reset_count()
            self.end_face_detection()
            self.end_mask_detection()
            self.end_grey_video()
            self.end_negative_video()
            self.end_horizontal_flip_video()
            self.end_vertical_flip_video()
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

    # Face Detection
    
    def face_detection_video(self):
        if self.video:
            if self.video.face_detection_is_enabled:
                self.end_face_detection()
            else:
                self.end_mask_detection()
                self.start_face_detection()
    
    def start_face_detection(self):
        self.video.face_detection_is_enabled = True
        self.button_face_detect.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_face_detection(self):
        self.video.face_detection_is_enabled = False
        self.button_face_detect.config(fg="#151515", bg="#FFFFFF")
        self.reset_count()

    # Mask Detection

    def mask_detection_video(self):
        if self.video:
            if self.video.mask_detection_is_enabled:
                self.end_mask_detection()
            else:
                self.end_face_detection()
                self.start_mask_detection()
    
    def start_mask_detection(self):
        self.video.mask_detection_is_enabled = True
        self.button_mask_detect.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_mask_detection(self):
        self.video.mask_detection_is_enabled = False
        self.button_mask_detect.config(fg="#151515", bg="#FFFFFF")
        self.reset_count()
    
    # Grey Video Effect

    def grey_video(self):
        if self.video:
            if self.video.grey_effect_is_enabled:
                self.end_grey_video()
            else:
                self.start_grey_video()
    
    def start_grey_video(self):
        self.video.grey_effect_is_enabled = True
        self.button_grey.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_grey_video(self):
        self.video.grey_effect_is_enabled = False
        self.button_grey.config(fg="#151515", bg="#FFFFFF")
    
    # Negative Video Effect

    def negative_video(self):
        if self.video:
            if self.video.negative_effect_is_enabled:
                self.end_negative_video()
            else:
                self.start_negative_video()
    
    def start_negative_video(self):
        self.video.negative_effect_is_enabled = True
        self.button_negative.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_negative_video(self):
        self.video.negative_effect_is_enabled = False
        self.button_negative.config(fg="#151515", bg="#FFFFFF")
    
    # Horizontal Flip Video Effect
    
    def horizontal_flip_video(self):
        if self.video:
            if self.video.horizontal_flip_effect_is_enabled:
                self.end_horizontal_flip_video()
            else:
                self.start_horizontal_flip_video()

    def start_horizontal_flip_video(self):
        self.video.horizontal_flip_effect_is_enabled = True
        self.button_horizontal_flip.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_horizontal_flip_video(self):
        self.video.horizontal_flip_effect_is_enabled = False
        self.button_horizontal_flip.config(fg="#151515", bg="#FFFFFF")
    
    # Vetical Flip Video Effect
    
    def vertical_flip_video(self):
        if self.video:
            if self.video.vertical_flip_effect_is_enabled:
                self.end_vertical_flip_video()
            else:
                self.start_vertical_flip_video()
    
    def start_vertical_flip_video(self):
        self.video.vertical_flip_effect_is_enabled = True
        self.button_vertical_flip.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_vertical_flip_video(self):
        self.video.vertical_flip_effect_is_enabled = False
        self.button_vertical_flip.config(fg="#151515", bg="#FFFFFF")

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
        self.horizontal_flip_effect_is_enabled = False
        self.vertical_flip_effect_is_enabled = False
        self.recording = False
        self.out = None
        self.record_frame = None

    def get_frame(self):
        if self.vid.isOpened():
            available, frame = self.vid.read()
            if available:
                
                if self.grey_effect_is_enabled:
                    frame = grey(frame)
                if self.negative_effect_is_enabled:
                    frame = negative(frame)
                if self.horizontal_flip_effect_is_enabled:
                    frame = horizontal_flip(frame)
                if self.vertical_flip_effect_is_enabled:
                    frame = vertical_flip(frame)
                if self.face_detection_is_enabled:
                    frame = detect_face(frame)
                if self.mask_detection_is_enabled:
                    frame = detect_mask(frame)

                frame = make_square(frame)
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                
                if self.recording:
                    self.record_frame = frame
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
        self.configure(background="#000C18")

        self.image = None

        self.image_video_replay = ImageTk.PhotoImage(Image.open("assets/images/video_replay.png"))
        self.image_video_snapshot = ImageTk.PhotoImage(Image.open("assets/images/video_snapshot.png"))
        self.image_file_open = ImageTk.PhotoImage(Image.open("assets/images/file_open.png"))
        self.image_video_blank = ImageTk.PhotoImage(Image.open("assets/images/image_blank.png"))
        
        self.canvas = tk.Canvas(self, width=498, height=498)
        self.canvas.pack(anchor="center", padx=20, pady=20)
        self.canvas.create_image(0, 0, image=self.image_video_blank, anchor='nw')
        
        self.image_buttons = tk.Frame(self, background="#000C18")
        self.image_buttons.pack()

        self.button_repeat = tk.Button(self.image_buttons, image=self.image_video_replay, command=self.restore_image, bd=0, background="#000C18", activebackground="#000C18")
        self.button_repeat.grid(row=1, column=1, padx=15, pady=15)

        self.button_snapshot = tk.Button(self.image_buttons, image=self.image_video_snapshot, command=self.take_snapshot, bd=0, background="#000C18", activebackground="#000C18")
        self.button_snapshot.grid(row=1, column=5, padx=15, pady=15)

        self.button_open = tk.Button(self.image_buttons, image=self.image_file_open, command=self.open_file, bd=0, background="#000C18", activebackground="#000C18")
        self.button_open.grid(row=1, column=6, padx=15, pady=15)
        
        self.image_button_effects = tk.Frame(self, background="#000C18")
        self.image_button_effects.pack()

        self.button_face_detect = tk.Button(self.image_button_effects, text="Face Detect", command=self.face_detection_image, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_face_detect.grid(row=1, column=0, padx=15, pady=15)

        self.button_mask_detect = tk.Button(self.image_button_effects, text="Mask Detect", command=self.mask_detection_image, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_mask_detect.grid(row=1, column=1, padx=15, pady=15)

        self.button_grey = tk.Button(self.image_button_effects, text="Grey", command=self.grey_image, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_grey.grid(row=1, column=2, padx=15, pady=15)

        self.button_negative = tk.Button(self.image_button_effects, text="Negative", command=self.negative_image, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_negative.grid(row=1, column=3, padx=15, pady=15)

        self.button_horizontal_flip = tk.Button(self.image_button_effects, text="H-Flip", command=self.horizontal_flip_image, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_horizontal_flip.grid(row=1, column=4, padx=15, pady=15)

        self.button_vertical_flip = tk.Button(self.image_button_effects, text="V-Flip", command=self.vertical_flip_image, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_vertical_flip.grid(row=1, column=5, padx=15, pady=15)

        self.button_back = tk.Button(self, text="Back", command=self.end_page, width=12, fg="#FFFFFF", bg="#E62A32", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12, "bold"), relief="raised")
        self.button_back.place(bordermode="outside", x=35, y=20)

        self.display_info()

        self.info_time = tk.Frame(self, background="#000C18")
        self.info_time.place(bordermode="outside", x=800, y=450)
        
        self.label_date = tk.Label(self.info_time, text=time.strftime('%B %d, %Y'),  font=("Tw Cen MT Condensed", 15), bg="#000C18", fg="#FFFFFF")
        self.label_date.grid(row=0, column=0, padx=5, pady=3, sticky="W")

        self.label_time = tk.Label(self.info_time, text=time.strftime('%H:%M %p'),  font=("Tw Cen MT Condensed", 13), bg="#000C18", fg="#FFFFFF")
        self.label_time.grid(row=1, column=0, padx=5, pady=3, sticky="W")

        self.update_time()
    
    def update_time(self):
        self.label_date.configure(text=time.strftime('%B %d, %Y'))
        self.label_time.configure(text=time.strftime('%H:%M:%S %p'))
        self.label_time.after(1000, self.update_time)
    
    def display_info(self):
        self.info = tk.Frame(self, background="#000C18")
        self.info.place(bordermode="outside", x=800, y=15)

        self.label_face_detected = tk.Label(self.info, text="Face Detected",  font=("Tw Cen MT Condensed", 12), bg="#000C18", fg="#FFFFFF")
        self.label_face_detected.grid(row=1, column=0, padx=5, pady=5, sticky="W")

        self.count_face_detected = tk.Label(self.info, text=face_detected_count,  font=("Tw Cen MT Condensed", 12, "bold"), bg="#000C18", fg="#FFFFFF")
        self.count_face_detected.grid(row=1, column=1, padx=5, pady=5, sticky="W")
        
        self.label_masked_detected = tk.Label(self.info, text="Masked Detected", font=("Tw Cen MT Condensed", 12), bg="#000C18", fg="#FFFFFF")
        self.label_masked_detected.grid(row=2, column=0, padx=5, pady=5, sticky="W")

        self.count_masked_detected = tk.Label(self.info, text=masked_detected_count, font=("Tw Cen MT Condensed", 12, "bold"), bg="#000C18", fg="#FFFFFF")
        self.count_masked_detected.grid(row=2, column=1, padx=5, pady=5, sticky="W")

        self.label_unmasked_detected = tk.Label(self.info, text="Unmasked Detected", font=("Tw Cen MT Condensed", 12), bg="#000C18", fg="#FFFFFF")
        self.label_unmasked_detected.grid(row=3, column=0, padx=5, pady=5, sticky="W")

        self.count_unmasked_detected = tk.Label(self.info, text=umasked_detected_count, font=("Tw Cen MT Condensed", 12, "bold"), bg="#000C18", fg="#FFFFFF")
        self.count_unmasked_detected.grid(row=3, column=1, padx=5, pady=5, sticky="W")
    
    def update_info(self):
        self.info.destroy()
        self.display_info()
    
    def reset_count(self):
        global face_detected_count, masked_detected_count, umasked_detected_count
        face_detected_count = 0
        masked_detected_count = 0
        umasked_detected_count = 0

    def end_page(self):
        self.destroy()
    
    def take_snapshot(self):
        try:
            if self.image:
                frame = self.image.get_frame()
                cv2.imwrite("snapshots/image-" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except:
            pass
    
    def display_image(self):
        if self.image:
            frame = self.image.get_frame()
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
    
    def open_file(self):
        self.image_filename = filedialog.askopenfilename(title="Open file", filetypes=(("JPG files", "*.jpg"), ("PNG files", "*.png")))
        if self.image_filename:
            self.image = ImageCapture(self.image_filename)
            self.restore_image()

    # Face Detection
    
    def face_detection_image(self):
        if self.image:
            if self.image.face_detection_is_enabled:
                self.end_face_detection()
            else:
                self.end_mask_detection()
                self.start_face_detection()
        self.display_image()
    
    def start_face_detection(self):
        self.image.face_detection_is_enabled = True
        self.button_face_detect.config(fg="#FFFFFF", bg="#00AAEB")
        self.update_info()
    
    def end_face_detection(self):
        self.image.face_detection_is_enabled = False
        self.button_face_detect.config(fg="#151515", bg="#FFFFFF")
        self.reset_count()

    # Mask Detection

    def mask_detection_image(self):
        if self.image:
            if self.image.mask_detection_is_enabled:
                self.end_mask_detection()
            else:
                self.end_face_detection()
                self.start_mask_detection()
        self.display_image()
    
    def start_mask_detection(self):
        self.image.mask_detection_is_enabled = True
        self.button_mask_detect.config(fg="#FFFFFF", bg="#00AAEB")
        self.update_info()
    
    def end_mask_detection(self):
        self.image.mask_detection_is_enabled = False
        self.button_mask_detect.config(fg="#151515", bg="#FFFFFF")
        self.reset_count()
    
    # Grey Image Effect

    def grey_image(self):
        if self.image:
            if self.image.grey_effect_is_enabled:
                self.end_grey_image()
            else:
                self.start_grey_image()
        self.display_image()
    
    def start_grey_image(self):
        self.image.grey_effect_is_enabled = True
        self.button_grey.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_grey_image(self):
        self.image.grey_effect_is_enabled = False
        self.button_grey.config(fg="#151515", bg="#FFFFFF")
    
    # Negative Image Effect

    def negative_image(self):
        if self.image:
            if self.image.negative_effect_is_enabled:
                self.end_negative_image()
            else:
                self.start_negative_image()
        self.display_image()
    
    def start_negative_image(self):
        self.image.negative_effect_is_enabled = True
        self.button_negative.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_negative_image(self):
        self.image.negative_effect_is_enabled = False
        self.button_negative.config(fg="#151515", bg="#FFFFFF")
    
    # Horizontal Flip Image Effect
    
    def horizontal_flip_image(self):
        if self.image:
            if self.image.horizontal_flip_effect_is_enabled:
                self.end_horizontal_flip_image()
            else:
                self.start_horizontal_flip_image()
        self.display_image()

    def start_horizontal_flip_image(self):
        self.image.horizontal_flip_effect_is_enabled = True
        self.button_horizontal_flip.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_horizontal_flip_image(self):
        self.image.horizontal_flip_effect_is_enabled = False
        self.button_horizontal_flip.config(fg="#151515", bg="#FFFFFF")
    
    # Vetical Flip Image Effect
    
    def vertical_flip_image(self):
        if self.image:
            if self.image.vertical_flip_effect_is_enabled:
                self.end_vertical_flip_image()
            else:
                self.start_vertical_flip_image()
        self.display_image()
    
    def start_vertical_flip_image(self):
        self.image.vertical_flip_effect_is_enabled = True
        self.button_vertical_flip.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_vertical_flip_image(self):
        self.image.vertical_flip_effect_is_enabled = False
        self.button_vertical_flip.config(fg="#151515", bg="#FFFFFF")
    
    def restore_image(self):
        self.end_face_detection()
        self.end_mask_detection()
        self.end_grey_image()
        self.end_negative_image()
        self.end_horizontal_flip_image()
        self.end_vertical_flip_image()

        self.display_image()

class ImageCapture:
    def __init__(self, source=None):
        self.source = source
        self.width = 500
        self.height = 500

        self.face_detection_is_enabled = False
        self.mask_detection_is_enabled = False
        self.negative_effect_is_enabled = False
        self.grey_effect_is_enabled = False
        self.horizontal_flip_effect_is_enabled = False
        self.vertical_flip_effect_is_enabled = False
        
    def get_frame(self):
        frame = cv2.imread(self.source)
        
        if self.grey_effect_is_enabled:
            frame = grey(frame)
        if self.negative_effect_is_enabled:
            frame = negative(frame)
        if self.horizontal_flip_effect_is_enabled:
            frame = horizontal_flip(frame)
        if self.vertical_flip_effect_is_enabled:
            frame = vertical_flip(frame)
        if self.face_detection_is_enabled:
            frame = detect_face(frame)
        if self.mask_detection_is_enabled:
            frame = detect_mask(frame)

        frame = make_square(frame)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class CameraPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(background="#000C18")
        
        self.video = VideoCapture(0)
        self.video_frame = None
        
        self.video_pause = True
        self.video_delay = 1
        self.video_end = False

        self.image_video_pause = ImageTk.PhotoImage(Image.open("assets/images/video_camera_off.png"))
        self.image_video_play = ImageTk.PhotoImage(Image.open("assets/images/video_camera_on.png"))
        self.image_video_loop_off = ImageTk.PhotoImage(Image.open("assets/images/video_loop_off.png"))
        self.image_video_loop_on = ImageTk.PhotoImage(Image.open("assets/images/video_loop_on.png"))
        self.image_video_replay = ImageTk.PhotoImage(Image.open("assets/images/video_replay.png"))
        self.image_video_record_off = ImageTk.PhotoImage(Image.open("assets/images/video_record_off.png"))
        self.image_video_record_on = ImageTk.PhotoImage(Image.open("assets/images/video_record_on.png"))
        self.image_file_open = ImageTk.PhotoImage(Image.open("assets/images/file_open.png"))
        self.image_video_snapshot = ImageTk.PhotoImage(Image.open("assets/images/video_snapshot.png"))
        self.image_video_blank = ImageTk.PhotoImage(Image.open("assets/images/camera_blank.png"))
        
        self.canvas = tk.Canvas(self, width=498, height=498)
        self.canvas.pack(anchor="center", padx=20, pady=20)
        self.canvas.create_image(0, 0, image=self.image_video_blank, anchor='nw')

        self.video_buttons = tk.Frame(self, background="#000C18")
        self.video_buttons.pack()

        self.button_pause = tk.Button(self.video_buttons, image=self.image_video_pause, command=self.switch_play, bd=0, background="#000C18", activebackground="#000C18")
        self.button_pause.grid(row=1, column=3, padx=15, pady=15)

        self.button_record = tk.Button(self.video_buttons, image=self.image_video_record_off, command=self.video_record, bd=0, background="#000C18", activebackground="#000C18")
        self.button_record.grid(row=1, column=4, padx=15, pady=15)

        self.button_snapshot = tk.Button(self.video_buttons, image=self.image_video_snapshot, command=self.take_snapshot, bd=0, background="#000C18", activebackground="#000C18")
        self.button_snapshot.grid(row=1, column=5, padx=15, pady=15)

        self.video_button_effects = tk.Frame(self, background="#000C18")
        self.video_button_effects.pack()

        self.button_face_detect = tk.Button(self.video_button_effects, text="Face Detect", command=self.face_detection_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_face_detect.grid(row=1, column=0, padx=15, pady=15)

        self.button_mask_detect = tk.Button(self.video_button_effects, text="Mask Detect", command=self.mask_detection_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_mask_detect.grid(row=1, column=1, padx=15, pady=15)

        self.button_grey = tk.Button(self.video_button_effects, text="Grey", command=self.grey_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_grey.grid(row=1, column=2, padx=15, pady=15)

        self.button_negative = tk.Button(self.video_button_effects, text="Negative", command=self.negative_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_negative.grid(row=1, column=3, padx=15, pady=15)

        self.button_horizontal_flip = tk.Button(self.video_button_effects, text="H-Flip", command=self.horizontal_flip_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_horizontal_flip.grid(row=1, column=4, padx=15, pady=15)

        self.button_vertical_flip = tk.Button(self.video_button_effects, text="V-Flip", command=self.vertical_flip_video, width=15, fg="#151515", bg="#FFFFFF", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12), relief="raised")
        self.button_vertical_flip.grid(row=1, column=5, padx=15, pady=15)

        self.button_back = tk.Button(self, text="Back", command=self.end_page, width=12, fg="#FFFFFF", bg="#E62A32", bd=0, activebackground="#15272F", activeforeground="#FFFFFF", font=("Tw Cen MT Condensed", 12, "bold"), relief="raised")
        self.button_back.place(bordermode="outside", x=35, y=20)

        self.info = tk.Frame(self, background="#000C18")
        self.info.place(bordermode="outside", x=800, y=15)

        self.label_face_detected = tk.Label(self.info, text="Face Detected",  font=("Tw Cen MT Condensed", 12), bg="#000C18", fg="#FFFFFF")
        self.label_face_detected.grid(row=1, column=0, padx=5, pady=5, sticky="W")

        self.count_face_detected = tk.Label(self.info, text=0,  font=("Tw Cen MT Condensed", 12, "bold"), bg="#000C18", fg="#FFFFFF")
        self.count_face_detected.grid(row=1, column=1, padx=5, pady=5, sticky="W")
        
        self.label_masked_detected = tk.Label(self.info, text="Masked Detected", font=("Tw Cen MT Condensed", 12), bg="#000C18", fg="#FFFFFF")
        self.label_masked_detected.grid(row=2, column=0, padx=5, pady=5, sticky="W")

        self.count_masked_detected = tk.Label(self.info, text=0, font=("Tw Cen MT Condensed", 12, "bold"), bg="#000C18", fg="#FFFFFF")
        self.count_masked_detected.grid(row=2, column=1, padx=5, pady=5, sticky="W")

        self.label_unmasked_detected = tk.Label(self.info, text="Unmasked Detected", font=("Tw Cen MT Condensed", 12), bg="#000C18", fg="#FFFFFF")
        self.label_unmasked_detected.grid(row=3, column=0, padx=5, pady=5, sticky="W")

        self.count_unmasked_detected = tk.Label(self.info, text=0, font=("Tw Cen MT Condensed", 12, "bold"), bg="#000C18", fg="#FFFFFF")
        self.count_unmasked_detected.grid(row=3, column=1, padx=5, pady=5, sticky="W")

        self.info_time = tk.Frame(self, background="#000C18")
        self.info_time.place(bordermode="outside", x=800, y=450)
        
        self.label_date = tk.Label(self.info_time, text=time.strftime('%B %d, %Y'),  font=("Tw Cen MT Condensed", 15), bg="#000C18", fg="#FFFFFF")
        self.label_date.grid(row=0, column=0, padx=5, pady=3, sticky="W")

        self.label_time = tk.Label(self.info_time, text=time.strftime('%H:%M %p'),  font=("Tw Cen MT Condensed", 13), bg="#000C18", fg="#FFFFFF")
        self.label_time.grid(row=1, column=0, padx=5, pady=3, sticky="W")

        self.update_time()
    
    def update_time(self):
        self.label_date.configure(text=time.strftime('%B %d, %Y'))
        self.label_time.configure(text=time.strftime('%H:%M:%S %p'))
        self.label_time.after(1000, self.update_time)
    
    def update_info(self):
        self.count_face_detected.configure(text=face_detected_count)
        self.count_masked_detected.configure(text=masked_detected_count)
        self.count_unmasked_detected.configure(text=umasked_detected_count)
    
    def reset_count(self):
        global face_detected_count, masked_detected_count, umasked_detected_count
        face_detected_count = 0
        masked_detected_count = 0
        umasked_detected_count = 0
    
    def end_page(self):
        self.reset_count()
        self.destroy()
    
    def take_snapshot(self):
        if self.video:
            try:
                cv2.imwrite(f"snapshots/image-{time.strftime('%Y-%m-%d-%H-%M-%S')}.jpg", cv2.cvtColor(self.video_frame, cv2.COLOR_RGB2BGR))
            except:
                pass
    
    def play_video(self):
        if self.video:
            self.update_info()
            if not self.video_pause:
                try:
                    self.video_frame = self.video.get_frame()
                    self.video_image = ImageTk.PhotoImage(image=Image.fromarray(self.video_frame))
                    self.canvas.create_image(0, 0, image=self.video_image, anchor='nw')
                except VideoRunOutOfFrame:
                    self.video_end = True
                    self.pause_video()
                    self.end_video_recording()
                self.after(self.video_delay, self.play_video)
            else:
                if self.video_end:
                    pass
                else:
                    self.reset_count()
                    self.end_face_detection()
                    self.end_mask_detection()
                    self.end_grey_video()
                    self.end_negative_video()
                    self.end_horizontal_flip_video()
                    self.end_vertical_flip_video()
                    self.canvas.create_image(0, 0, image=self.image_video_blank, anchor='nw')
    
    def switch_play(self):
        if self.video:
            if self.video_pause:
                self.resume_video()
            else:
                self.pause_video()
    
    def pause_video(self):
        self.video_pause = True
        self.button_pause.config(image=self.image_video_pause)
        self.end_video_recording()
    
    def resume_video(self):
        if self.video_pause:
            self.video_pause = False
            self.button_pause.config(image=self.image_video_play)
            self.play_video()
    
    def video_record(self):
        if self.video and not self.video_pause and not self.video_end:
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

   # Face Detection
    
    def face_detection_video(self):
        if self.video and not self.video_pause:
            if self.video.face_detection_is_enabled:
                self.end_face_detection()
            else:
                self.end_mask_detection()
                self.start_face_detection()
    
    def start_face_detection(self):
        self.video.face_detection_is_enabled = True
        self.button_face_detect.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_face_detection(self):
        self.video.face_detection_is_enabled = False
        self.button_face_detect.config(fg="#151515", bg="#FFFFFF")
        self.reset_count()

    # Mask Detection

    def mask_detection_video(self):
        if self.video and not self.video_pause:
            if self.video.mask_detection_is_enabled:
                self.end_mask_detection()
            else:
                self.end_face_detection()
                self.start_mask_detection()
    
    def start_mask_detection(self):
        self.video.mask_detection_is_enabled = True
        self.button_mask_detect.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_mask_detection(self):
        self.video.mask_detection_is_enabled = False
        self.button_mask_detect.config(fg="#151515", bg="#FFFFFF")
        self.reset_count()
    
    # Grey Video Effect

    def grey_video(self):
        if self.video and not self.video_pause:
            if self.video.grey_effect_is_enabled:
                self.end_grey_video()
            else:
                self.start_grey_video()
    
    def start_grey_video(self):
        self.video.grey_effect_is_enabled = True
        self.button_grey.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_grey_video(self):
        self.video.grey_effect_is_enabled = False
        self.button_grey.config(fg="#151515", bg="#FFFFFF")
    
    # Negative Video Effect

    def negative_video(self):
        if self.video and not self.video_pause:
            if self.video.negative_effect_is_enabled:
                self.end_negative_video()
            else:
                self.start_negative_video()
    
    def start_negative_video(self):
        self.video.negative_effect_is_enabled = True
        self.button_negative.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_negative_video(self):
        self.video.negative_effect_is_enabled = False
        self.button_negative.config(fg="#151515", bg="#FFFFFF")
    
    # Horizontal Flip Video Effect
    
    def horizontal_flip_video(self):
        if self.video and not self.video_pause:
            if self.video.horizontal_flip_effect_is_enabled:
                self.end_horizontal_flip_video()
            else:
                self.start_horizontal_flip_video()

    def start_horizontal_flip_video(self):
        self.video.horizontal_flip_effect_is_enabled = True
        self.button_horizontal_flip.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_horizontal_flip_video(self):
        self.video.horizontal_flip_effect_is_enabled = False
        self.button_horizontal_flip.config(fg="#151515", bg="#FFFFFF")
    
    # Vetical Flip Video Effect
    
    def vertical_flip_video(self):
        if self.video and not self.video_pause:
            if self.video.vertical_flip_effect_is_enabled:
                self.end_vertical_flip_video()
            else:
                self.start_vertical_flip_video()
    
    def start_vertical_flip_video(self):
        self.video.vertical_flip_effect_is_enabled = True
        self.button_vertical_flip.config(fg="#FFFFFF", bg="#00AAEB")
    
    def end_vertical_flip_video(self):
        self.video.vertical_flip_effect_is_enabled = False
        self.button_vertical_flip.config(fg="#151515", bg="#FFFFFF")


if __name__ == "__main__":
    app = tkinterApp()
    app.mainloop()