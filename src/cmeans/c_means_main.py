import cupy as cp
from utils import FuzzyDetector


class CMeans:
    def __init__(self, n_clusters:int=3,**kwargs):
        super.__init__()
        self.n_clusters = n_clusters
        self.fuzziness = 2.0
        self.metric = "euclidean"
        
        seed = kwargs.get("seed", None)
        init = kwargs.get("init", None)
        if seed:
            cp.random.seed(seed=seed)
        if init:
            self.init = init

    def cmeans(self,frame):
        if self.init:
            u0 = cp.array(self.init)
        else:
            n = frame.shape[1]
            u0 = cp.random.rand(self.n_clusters,n)
        
    def process_frame(self):
        # El frame es de 3 colores
        frame_color,color = self.frame.shape[0] * self.frame.shape[1], self.frame.shape[2]
        frame = cp.reshape(self.frame, (frame_color, color))
        