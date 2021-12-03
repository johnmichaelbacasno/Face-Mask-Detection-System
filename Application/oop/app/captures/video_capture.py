from .. functions import grey, negative, horizontal_flip, vertical_flip, make_square, detect_face, detect_mask

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