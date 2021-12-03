import tkinter as tk
from PIL import Image, ImageTk

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