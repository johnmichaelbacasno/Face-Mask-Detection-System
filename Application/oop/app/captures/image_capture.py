from .. functions import grey, negative, horizontal_flip, vertical_flip, make_square, detect_face, detect_mask

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