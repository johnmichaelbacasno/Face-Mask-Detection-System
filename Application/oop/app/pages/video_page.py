import tkinter as tk

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