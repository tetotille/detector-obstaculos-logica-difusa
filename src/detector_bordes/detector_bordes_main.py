import cupy as cp
from utils import FuzzyDetector


class FuzzyContour(FuzzyDetector):
    def __init__(self,image:cp.array):
        super().__init__()
        self.frame = image # imagen de 256xB en escala de grises
        self.kernel = {
            "membership_function": None,
            "apply_rule_kernel": None,
            "defuzzify": None
        }

    def apply_rules(self,windows):
        rules = cp.RawKernel(self.kernel["apply_rule_kernel"], "apply_rule_kernel")
        result = cp.zeros_like(windows)
        rules((self.GRID_SIZE,), (self.BLOCK_SIZE,), (windows, result))
        return result
    
    def defuzzify(self,fuzzy_matrix):
        defuzzify = cp.RawKernel(self.kernel["defuzzify"], "defuzzify_kernel")
        return defuzzify((self.GRID_SIZE,), (self.BLOCK_SIZE,), (fuzzy_matrix))

    def generate_windows(self,low,medium,high):
        height, width = self.frame.shape
        windows = cp.empty((height-1)*(width-1), dtype=cp.object)

        @cp.fuse()
        def generate_window(i, j, low, medium, high):
            return cp.array([low[i-1:i+1, j-1:j+1], medium[i-1:i+1, j-1:j+1], high[i-1:i+1, j-1:j+1]])

        for k in range((height-1)*(width-1)):
            i = k // (width-1)
            j = k % (width-1)
            windows[k] = generate_window(i, j, low, medium, high)
    
    # Este es como el main del objeto
    def process_frame(self):
        low,medium,high = self.membership_function()

        windows = self.generate_windows(low,medium,high)
        edge = {"low":low,"yes":lambda x: cp.full_like(x, cp.float32(0.5)), "high":high}

        result = self.apply_rules(windows)
        defuzzified = self.defuzzify(result)
        return defuzzified
        
    
    def membership_function(self):
        """
        Applies a membership function to the frame to classify pixel values into low, medium, and high categories.

        This function initializes output matrices for low, medium, and high membership values, and then uses a CUDA kernel
        to compute the membership values for each pixel in the frame.

        Returns:
            tuple: A tuple containing three arrays (low_output, medium_output, high_output) representing the membership
                   values for low, medium, and high categories respectively.
        """
        size = self.frame.size
        max_pixel = cp.max(self.frame)
        min_pixel = cp.min(self.frame)

        # Inicializacion de las matrices de salida
        low_output = cp.empty_like(self.frame)
        medium_output = cp.empty_like(self.frame)
        high_output = cp.empty_like(self.frame)

        apply_mf = cp.RawKernel(self.kernel["membership_function"], "membership_kernel")
        apply_mf((self.GRID_SIZE,), (self.BLOCK_SIZE,),
                 (self.frame, low_output, medium_output, high_output, size, max_pixel, min_pixel))

        return low_output, medium_output, high_output

if __name__ == "__main__":
    from os.path import join, dirname, abspath
    from src.utils import read_image

    filename = join(dirname(dirname(dirname(abspath(__file__)))), "assets/images/akaso1.jpeg")
    image = read_image(filename,256,grayscale=True,normalize=True)
