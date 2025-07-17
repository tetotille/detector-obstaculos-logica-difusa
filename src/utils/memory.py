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

    def get_point(self):
        return self._frames[self._last]["x_centroid"],self._frames[self._last]["y_centroid"]
    
    def get_weight(self):
        return self._frames[self._last]["weight"]
    
    def get_rectangle(self):
        return [self._frames[self._last]["x_init"],self._frames[self._last]["y_init"]],[self._frames[self._last]["x_end"],self._frames[self._last]["y_end"]]

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
        xi1 = self._frames[self._last]["x_init"]
        xi2 = frame_data["x_init"]
        xe1 = self._frames[self._last]["x_end"]
        xe2 = frame_data["x_end"]
        yi1 = self._frames[self._last]["y_init"]
        yi2 = frame_data["y_init"]
        ye1 = self._frames[self._last]["y_end"]
        ye2 = frame_data["y_end"]
        return not (xe1 < xi2 or xe2 < xi1 or ye1 < yi2 or ye2 < yi1)
    
    def __eq__(self,other):
        return self._id == other._id
    
    def __str__(self):
        points = sum([f"P{x}:{self._frames[x]['x_centroid'],self._frames[x]['y_centroid']}" if self._frames[x] is not None else None for x in range(self._cantidad)])
        return f"FrameMemory(id:{self._id},{points})"