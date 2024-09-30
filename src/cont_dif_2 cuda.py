import cv2
import matplotlib.pyplot as plt
import cupy as cp
import skfuzzy as fuzz

from os.path import dirname, abspath, join
from skfuzzy import control as ctrl
from skimage import io
from sys import argv

from fuzzy_detector import FuzzyDetector
from hsv_filter import filter_h,filter_s

class ContDifDetector(FuzzyDetector):
    def __init__(self,**kwargs):
        """
        Initializes a fuzzy logic control system.

        Parameters:
        **kwargs (dict): Optional keyword arguments.
            - min_pixel (int, optional): Minimum pixel value. Defaults to 300.
            - max_pixel (int, optional): Maximum pixel value. Defaults to 300.

        Attributes:
        - min_pixel (int): Minimum pixel value.
        - max_pixel (int): Maximum pixel value.
        - antecedent_universe (numpy.ndarray): Universe of discourse for antecedents.
        - consequent_universe (numpy.ndarray): Universe of discourse for consequent.
        - antecedents (list): List of Antecedent objects.
        - consequent (Consequent): Consequent object.
        - control_system (ControlSystem): Fuzzy logic control system.
        - control_system_simulation (ControlSystemSimulation): Control system simulation.

        Notes:
        - Calls `define_membership_functions` and `define_rules` methods to set up the fuzzy logic control system.
        """
        super().__init__()
        self.min_pixel = kwargs.get("min_pixel",300)
        self.max_pixel = kwargs.get("max_pixel",300)

        self.antecedent_universe = cp.arange(self.min_pixel, self.max_pixel, (1/256))
        self.consequent_universe = cp.arange(0, 256)/256

        self.antecedents = [ctrl.Antecedent(self.antecedent_universe,f'C{x}') for x in range(1,10)]
        self.consequent = ctrl.Consequent(self.consequent_universe, 'edge')
        
        self.define_membership_functions()
        self.define_rules()

        self.control_system = ctrl.ControlSystem(self.rules)
        self.control_system_simulation = ctrl.ControlSystemSimulation(self.control_system)

    def define_membership_functions(self):
        valor_medio = (int(self.min_pixel) + int(self.max_pixel))/2
        for C in self.antecedents:
            C['low'] = fuzz.trimf(self.antecedent_universe, [self.min_pixel, self.min_pixel, valor_medio])
            # C['medium'] = fuzz.trimf(self.antecedent_universe, [(self.max_pixel/3), valor_medio, self.max_pixel*(2/3)])
            C['medium'] = fuzz.gaussmf(self.antecedent_universe, valor_medio,valor_medio/3)
            C['high'] = fuzz.trimf(self.antecedent_universe, [max_pixel/2, self.max_pixel, self.max_pixel])
        
        self.consequent['low'] = fuzz.trimf(self.consequent_universe, [0, 0, 0.6])
        self.consequent['high']= fuzz.trimf(self.consequent_universe, [0.5, 1, 1])
        self.consequent['yes'] = fuzz.trimf(self.consequent_universe, [0.4, 0.5, 0.6])

    def show_model(self):
        # Graficar las funciones de membresía
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Funciones de membresía para C (low y high)
        ax[0].plot(self.antecedent_universe, self.antecedents[0]["low"].mf, 'b', linewidth=1.5, label='Low')
        ax[0].plot(self.antecedent_universe, self.antecedents[0]["medium"].mf, 'g', linewidth=1.5, label='Medium')
        ax[0].plot(self.antecedent_universe, self.antecedents[0]["high"].mf, 'r', linewidth=1.5, label='High')
        ax[0].set_title('Funciones de Membresía para C')
        ax[0].legend()

        # Funciones de membresía para edge (no y yes)
        ax[1].plot(self.consequent_universe, self.consequent["low"].mf, 'b', linewidth=1.5, label='No')
        ax[1].plot(self.consequent_universe, self.consequent["yes"].mf, 'r', linewidth=1.5, label='Yes')
        ax[1].plot(self.consequent_universe, self.consequent["high"].mf, 'y', linewidth=1.5, label='High')   
        ax[1].set_title('Funciones de Membresía para Edge')
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    def define_rules(self):
        C1,C2,C3,C4,C5,C6,C7,C8,C9 = self.antecedents
        edge = self.consequent
        rules = []

        ######## BLOQUES HIGH #########
        # Bloque 1 hor,ver, 1 dimension
        rules.append(ctrl.Rule( C1['high'] & C2['high'] & C3['high'] &  # 0
                                C4['low']  & C5['low']  & C6['low'] & 
                                C7['low']  & C8['low']  & C9['low'], edge['yes']))
        
        rules.append(ctrl.Rule( C1['high'] & C2['low']  & C3['low'] &  # 1
                                C4['high'] & C5['low']  & C6['low'] & 
                                C7['high'] & C8['low']  & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']  & C3['low'] &  # 2
                                C4['low']  & C5['low']  & C6['low'] & 
                                C7['high'] & C8['high'] & C9['high'], edge['yes']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']  & C3['high'] & # 3
                                C4['low']  & C5['low']  & C6['high'] & 
                                C7['low']  & C8['low']  & C9['high'], edge['high']))
        
        # Bloque 2 hor, ver, 2 dimensiones
        rules.append(ctrl.Rule( C1['high'] & C2['high'] & C3['high'] &  # 4
                                C4['high'] & C5['high'] & C6['high'] & 
                                C7['low']  & C8['low']  & C9['low'], edge['yes']))
        
        rules.append(ctrl.Rule( C1['high'] & C2['high']  & C3['low'] &  # 5
                                C4['high'] & C5['high']  & C6['low'] & 
                                C7['high'] & C8['high']  & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']   & C3['low'] &  # 6
                                C4['high'] & C5['high']  & C6['high'] & 
                                C7['high'] & C8['high']  & C9['high'], edge['yes']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['high']  & C3['high'] & # 7
                                C4['low']  & C5['high']  & C6['high'] & 
                                C7['low']  & C8['high']  & C9['high'], edge['high']))
        
        # Bloque 3 esquinas grandes
        rules.append(ctrl.Rule( C1['high'] & C2['high']  & C3['high'] &  # 8
                                C4['low']  & C5['high']  & C6['high'] & 
                                C7['low']  & C8['low']   & C9['high'], edge['high']))
        
        rules.append(ctrl.Rule( C1['high'] & C2['high']  & C3['high'] &  # 9
                                C4['high'] & C5['high']  & C6['low'] & 
                                C7['high'] & C8['low']   & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( C1['high'] & C2['low']   & C3['low'] &  # 10
                                C4['high'] & C5['high']  & C6['low'] & 
                                C7['high'] & C8['high']  & C9['high'], edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']   & C3['high'] & # 11
                                C4['low']  & C5['high']  & C6['high'] & 
                                C7['high'] & C8['high']  & C9['high'], edge['high']))
        
        # Bloque 4 esquinas cuadradas
        rules.append(ctrl.Rule( C1['low']  & C2['high'] & C3['high'] &  # 12
                                C4['low']  & C5['high'] & C6['high'] & 
                                C7['low']  & C8['low']  & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']  & C3['low'] &  # 13
                                C4['low']  & C5['high'] & C6['high'] & 
                                C7['low']  & C8['high'] & C9['high'], edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']   & C3['low'] &  # 14
                                C4['high'] & C5['high']  & C6['low'] & 
                                C7['high'] & C8['high']  & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( C1['high'] & C2['high']  & C3['low'] & # 15
                                C4['high'] & C5['high']  & C6['low'] & 
                                C7['low']  & C8['low']   & C9['low'], edge['high']))
        
        ##### BLOQUES YES ######
        # Bloque 1 hor,ver, 1 dimension
        rules.append(ctrl.Rule( (C1['high'] | C1["medium"]) & (C2['high'] | C2["medium"]) & (C3['high'] | C3["medium"]) &  # 0
                                C4['low']  & C5['low']  & C6['low'] & 
                                C7['low']  & C8['low']  & C9['low'], edge['low']))
        
        rules.append(ctrl.Rule( (C1['high'] | C1["medium"]) & C2['low']  & C3['low'] &  # 1
                                (C4['high'] | C4["medium"]) & C5['low']  & C6['low'] & 
                                (C7['high'] | C7["medium"]) & C8['low']  & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']  & C3['low'] &  # 2
                                C4['low']  & C5['low']  & C6['low'] & 
                                (C7['high'] | C7["medium"]) & (C8['high'] | C8["medium"]) & (C9['high'] | C9["medium"]), edge['low']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']  & (C3['high'] | C3["medium"]) & # 3
                                C4['low']  & C5['low']  & (C6['high'] | C6["medium"]) & 
                                C7['low']  & C8['low']  & (C9['high'] | C9["medium"]), edge['high']))
        
        # Bloque 2 hor, ver, 2 dimensiones
        rules.append(ctrl.Rule( (C1['high'] | C1["medium"]) & (C2['high'] | C2["medium"]) & (C3['high'] | C3["medium"]) &  # 4
                                (C4['high'] | C4["medium"]) & (C5['high'] | C5["medium"]) & (C6['high'] | C6["medium"]) & 
                                C7['low']  & C8['low']  & C9['low'], edge['low']))
        
        rules.append(ctrl.Rule( (C1['high'] | C1["medium"]) & (C2['high'] | C2["medium"])  & C3['low'] &  # 5
                                (C4['high'] | C4["medium"]) & (C5['high'] | C5["medium"])  & C6['low'] & 
                                (C7['high'] | C7["medium"]) & (C8['high'] | C8["medium"])  & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']   & C3['low'] &  # 6
                                (C4['high'] | C4["medium"]) & (C5['high'] | C5["medium"])  & (C6['high'] | C6["medium"]) & 
                                (C7['high'] | C7["medium"]) & (C8['high'] | C8["medium"])  & (C9['high'] | C9["medium"]), edge['low']))
        
        rules.append(ctrl.Rule( C1['low']  & (C2['high'] | C2["medium"])  & (C3['high'] | C3["medium"]) & # 7
                                C4['low']  & (C5['high'] | C5["medium"])  & (C6['high'] | C6["medium"]) & 
                                C7['low']  & (C8['high'] | C8["medium"])  & (C9['high'] | C9["medium"]), edge['high']))
        
        # Bloque 3 esquinas grandes
        rules.append(ctrl.Rule( (C1['high'] | C1["medium"]) & (C2['high'] | C2["medium"])  & (C3['high'] | C3["medium"]) &  # 8
                                C4['low']  & (C5['high'] | C5["medium"])  & (C6['high'] | C6["medium"]) & 
                                C7['low']  & C8['low']   & (C9['high'] | C9["medium"]), edge['high']))
        
        rules.append(ctrl.Rule( (C1['high'] | C1["medium"]) & (C2['high'] | C2["medium"])  & (C3['high'] | C3["medium"]) &  # 9
                                (C4['high'] | C4["medium"]) & (C5['high'] | C5["medium"])  & C6['low'] & 
                                (C7['high'] | C7["medium"]) & C8['low']   & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( (C1['high'] | C1["medium"]) & C2['low']   & C3['low'] &  # 10
                                (C4['high'] | C4["medium"]) & (C5['high'] | C5["medium"])  & C6['low'] & 
                                (C7['high'] | C7["medium"]) & (C8['high'] | C8["medium"])  & (C9['high'] | C9["medium"]), edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']   & (C3['high'] | C3["medium"]) & # 11
                                C4['low']  & (C5['high'] | C5["medium"])  & (C6['high'] | C6["medium"]) & 
                                (C7['high'] | C7["medium"]) & (C8['high'] | C8["medium"])  & (C9['high'] | C9["medium"]), edge['high']))
        
        # Bloque 4 esquinas cuadradas
        rules.append(ctrl.Rule( C1['low']  & (C2['high'] | C2["medium"]) & (C3['high'] | C3["medium"]) &  # 12
                                C4['low']  & (C5['high'] | C5["medium"]) & (C6['high'] | C6["medium"]) & 
                                C7['low']  & C8['low']  & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']  & C3['low'] &  # 13
                                C4['low']  & (C5['high'] | C5["medium"]) & (C6['high'] | C6["medium"]) & 
                                C7['low']  & (C8['high'] | C8["medium"]) & (C9['high'] | C9["medium"]), edge['high']))
        
        rules.append(ctrl.Rule( C1['low']  & C2['low']   & C3['low'] &  # 14
                                (C4['high'] | C4["medium"]) & (C5['high'] | C5["medium"])  & C6['low'] & 
                                (C7['high'] | C7["medium"]) & (C8['high'] | C8["medium"])  & C9['low'], edge['high']))
        
        rules.append(ctrl.Rule( (C1['high'] | C1["medium"]) & (C2['high'] | C2["medium"])  & C3['low'] & # 15
                                (C4['high'] | C4["medium"]) & (C5['high'] | C5["medium"])  & C6['low'] & 
                                C7['low']  & C8['low']   & C9['low'], edge['high']))
        
        
        self.rules=rules
        
    def fuzzify(self,input_values:list):
        for i in range(len(input_values)):
            self.control_system_simulation.input[f"C{i+1}"] = input_values[i]

    def defuzzify(self):
        self.control_system_simulation.compute()
        return self.control_system_simulation.output["edge"]
    
    def process_image(self,image,show=False):
        output_image = np.zeros_like(image)
        n,m = image.shape

        neighbors = [(-1, -1), (-1, 0), (-1, 1), 
                     (0, -1),  (0, 0),  (0, 1),  
                     (1, -1),  (1, 0),  (1, 1)]
        
        for i in range(1,n-1):
            for j in range(1,m-1):
                neighbor_values = [image[i+di,j+dj] for di, dj in neighbors]
                self.fuzzify(neighbor_values)
                try:
                    output_image[i,j] = self.defuzzify()
                    print("Se cumplió", output_image[i,j])
                except Exception as e:
                    # print(e)
                    output_image[i,j] = 0
            print(f"Procesando: {int(((i)*(m)/(n*m))*100)} %   ",end="\r")

        if show:
            io.imshow(output_image)
            io.show()

        return output_image

# Función para cargar una imagen desde la computadora
def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

if __name__ == "__main__":
    if len(argv) > 1:
        filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
    else:
        filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")

    image=cv2.imread(filename)
    x,y,a = image.shape
    #image2=filter_h(image)
    imagen=cv2.resize(image, (200, int(x*200/y)))
    gray=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    


    # Aplicar detección de bordes difusa
    fuzzy_image = filter_h(imagen)
    cv2.imshow('Original Image', fuzzy_image)
    cv2.waitKey(0)
    
    min_pixel = fuzzy_image.min()
    max_pixel = fuzzy_image.max()

    detector = ContDifDetector(min_pixel=min_pixel,max_pixel=max_pixel)
    detector.show_model()
    output_image = detector.process_image(fuzzy_image.astype(float))

    edge_image_uint8:np.array = (output_image * 255).astype(np.uint8)

    

    # Aplicar el umbral negro
    _, imagen_umbral = cv2.threshold(edge_image_uint8, 1, 255, cv2.THRESH_BINARY)

    # Suavizar la máscara de contornos
    mask_smooth = cv2.GaussianBlur(imagen_umbral, (5, 5), 0)


    if np.mean(mask_smooth) > max_pixel/2:
        mask_smooth = 255 - mask_smooth

    # Mostrar resultados
    cv2.imshow('Original Image', imagen)
    cv2.imshow('Filtered Image', fuzzy_image)
    cv2.imshow('Solo contorno', edge_image_uint8)
    cv2.imshow('Fuzzy Edge Detected Image', mask_smooth)  # Escalar a 0-255 para visualizar
    cv2.waitKey(0)
    cv2.imwrite("tes.jpeg",imagen_umbral)
    cv2.imwrite("filename.png", imagen_umbral)
    cv2.destroyAllWindows()

    