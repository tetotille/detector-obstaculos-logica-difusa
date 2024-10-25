from abc import ABC, abstractmethod

class FuzzyDetector(ABC):
    def __init__(self):
        self.kernel = dict()
        self.frame = None
        self.result = None
        self.BLOCK_SIZE = 256
        self.GRID_SIZE = (256*256*3+self.BLOCK_SIZE-1)

    def update_frame(self, frame):
        self.frame = frame
        self.result = self.process_frame()

    def update_kernel(self, name, kernel):
        self.kernel[name] = kernel

    def show_results(self):
        if self.result is None:
            print("Se procesará el frame actual.")
            self.process_frame()
        return self.result

    @abstractmethod
    def membership_function(self):
        raise NotImplementedError

    @abstractmethod
    def apply_rules(self):
        raise NotImplementedError
    
    @abstractmethod
    def defuzzify(self):
        raise NotImplementedError
    
    @abstractmethod
    def fuzzify(self, input_value):
        raise NotImplementedError

    @abstractmethod
    def process_frame(self):
        raise NotImplementedError
    
    @abstractmethod
    def show_model(self):
        raise NotImplementedError
    
