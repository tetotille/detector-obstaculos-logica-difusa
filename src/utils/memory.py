import uuid

class FrameMemory:
    def __init__(self,cantidad:int,frame_data):
        self._id = uuid.uuid4()
        self._cantidad = cantidad
        self._frames = [None]*cantidad
        self._last = 0
        self._index = 0
        self._insert(frame_data,0)
        self.modified = False
        
    def add(self,frame_data):
        index = self._index + 1
        index = 0 if index == self._cantidad else index
        self._insert(frame_data,index)

    def empty(self):
        return self._frames[self._last] is None

    def pop(self,index):
        result = self._frames[index]
        self._frames[index] = None
        return result

    def score(self):
        return sum(1 for frame in self._frames if frame is not None)
    
    def _insert(self,frame_data,index):
        if frame_data is None:
            self._frames[index] = None
            self._index = index
        else:
            self._frames[index] = {
                "x_init": frame_data["x_init"],
                "x_end": frame_data["x_end"],
                "y_init": frame_data["y_init"],
                "y_end": frame_data["y_end"],
                "x_centroid": frame_data["x_centroid"],
                "y_centroid": frame_data["y_centroid"],
                "weight": frame_data["weight"],
                "fuzzy_union": frame_data["fuzzy_union"]
            }
            self._last = index
            self._index = index

    def __contains__(self,frame_data):
        return (frame_data["x_init"] >= self._frames[self._last]["x_init"]) and (frame_data["x_end"] <= self._frames[self._last]["x_end"]) and (frame_data["y_init"] >= self._frames[self._last]["y_init"]) and (frame_data["y_end"] <= self._frames[self._last]["y_end"])

    def __eq__(self,other):
        return self._id == other._id