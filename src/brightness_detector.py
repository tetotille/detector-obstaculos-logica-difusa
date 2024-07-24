import numpy as np
import os
import pyopencl as cl
import skfuzzy as fuzz

from abc import ABC, abstractmethod
from os.path import abspath,dirname,join
from skfuzzy import control as ctrl
from skimage import io
from sys import argv

os.environ["PYOPENCL_ICD_KHR"] = "/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"

# Si hay mucho brillo es probable que el objeto salga oscuro

class FuzzyDetector(ABC):
    def __init__(self):
        self.antecedent = None
        self.consequent = None
        self.rules = []
        self.control_system = None
        self.control_system_simulation = None
        self.kernel = None

    @abstractmethod
    def define_membership_functions(self):
        pass

    @abstractmethod
    def define_rules(self):
        pass

    @abstractmethod
    def fuzzify(self, input_value):
        pass

    @abstractmethod
    def defuzzify(self):
        pass

class BrightnessDetector(FuzzyDetector):
    def __init__(self):
        super().__init__()
        self.antecedent = ctrl.Antecedent(np.arange(0, 1, 0.01), 'brightness')
        self.consequent = ctrl.Consequent(np.arange(0, 256, 1), 'darkness')
        self.define_membership_functions()
        self.define_rules()
        self.control_system = ctrl.ControlSystem(self.rules)
        self.control_system_simulation = ctrl.ControlSystemSimulation(self.control_system)

        self.kernel = """
                        __kernel void process_image(__global const float *image, __global float *output, int width, int height) {
                            int i = get_global_id(0);
                            int j = get_global_id(1);
                            if (i < height && j < width) {
                                // Fuzzify the input value
                                float brightness = image[i * width + j];
                                float dark = 0.0f;
                                float partially_dark = 0.0f;
                                float not_dark = 0.0f;
                                if (brightness <= 0.5f) {
                                    dark = 1.0f - brightness * 2.0f;
                                    partially_dark = brightness * 2.0f;
                                } else {
                                    partially_dark = 2.0f - brightness * 2.0f;
                                    not_dark = brightness * 2.0f - 1.0f;
                                }

                                // Apply the rules
                                float dark_output = fminf(dark, 1.0f - brightness);
                                float partially_dark_output = fminf(partially_dark, fminf(brightness, 1.0f - brightness));
                                float not_dark_output = fminf(not_dark, brightness);

                                // Defuzzify the output value
                                float output_value = (dark_output * 0.0f + partially_dark_output * 100.0f + not_dark_output * 255.0f) / (dark_output + partially_dark_output + not_dark_output);

                                // Write the output value to the output image
                                output[i * width + j] = output_value;
                            }
                        }
                    """

    def define_membership_functions(self):
        self.antecedent['dark'] = fuzz.trimf(self.antecedent.universe, [0, 30/255, 60/255])
        self.antecedent['partially_dark'] = fuzz.trimf(self.antecedent.universe, [40/255, 60/255, 80/255])
        self.antecedent['not_dark'] = fuzz.trimf(self.antecedent.universe, [70/255, 110/255, 1])

        self.consequent['dark'] = fuzz.trimf(self.consequent.universe, [0, 30, 60])
        self.consequent['partially_dark'] = fuzz.trimf(self.consequent.universe, [40, 60, 80])
        self.consequent['not_dark'] = fuzz.trimf(self.consequent.universe, [70, 110, 255])

    def define_rules(self):
        rule1 = ctrl.Rule(self.antecedent['dark'], self.consequent['dark'])
        rule2 = ctrl.Rule(self.antecedent['partially_dark'], self.consequent['partially_dark'])
        rule3 = ctrl.Rule(self.antecedent['not_dark'], self.consequent['not_dark'])
        self.rules = [rule1, rule2, rule3]

    def fuzzify(self, input_value):
        self.control_system_simulation.input['brightness'] = input_value

    def defuzzify(self):
        self.control_system_simulation.compute()
        output = self.control_system_simulation.output['darkness']
        if output <= 50:
            return 0
        elif output >= 150:
            return 255
        else:
            return 100

    def process_image(self, image):
        # Process the image
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

         # Allocate memory for the input and output images on the device
        mf = cl.mem_flags
        image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, image.nbytes)

        prg = cl.Program(ctx,self.kernel).build()

        prg.process_image(queue, image.shape, None, image_buf, output_buf, np.int32(image.shape[1]), np.int32(image.shape[0]))

        # Read the output image from the device
        output_image = np.empty_like(image)
        cl.enqueue_copy(queue, output_image, output_buf)


        # output_image = np.zeros_like(image)
        # n,m = image.shape
        # for i in range(n):
        #     for j in range(m):
        #         self.fuzzify(image[i, j])
        #         output_image[i, j] = self.defuzzify()
        #     print(f"Procesando: {int(((i)*(m)/(n*m))*100)} %   ",end="\r")

        # Save the output image
        io.imshow(output_image)
        io.show()

if __name__ == "__main__":
    if len(argv) <= 1: raise(NameError("You have to enter a file name as an argument"))
    if os.path.dirname(argv[1]):
        image = io.imread(argv[1],as_gray=True)
    else:
        img_path = join(dirname(dirname(abspath(__file__))),f"img/{argv[1]}")
        image = io.imread(img_path,as_gray=True)

    if len(argv) >= 3 and argv[2] == "mostrar":
        io.imshow(image)
        io.show()

    detector = BrightnessDetector()
    detector.process_image(image)

    

