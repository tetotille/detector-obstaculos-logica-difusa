class ObjectDetector:
    def __init__(self,frame):
        self.frame = frame
        self.position = (None,None,)
        self.left_grade = 0.0
        self.center_grade = 0.0
        self.right_grade = 0.0
        self.centroid = [0,0]
        self.object_detected = False

    def update_frame(self,frame):
        self.frame = frame

    def get_membership(self):
        return {"left":self.left_grade,"center":self.center_grade,"right":self.right_grade}
    
    def get_detection(self):
        return self.object_detected
    
    def get_centroid(self):
        return self.centroid
    
    def get_position(self):
        return self.position
    
    async def detect(self):
        raise NotImplementedError