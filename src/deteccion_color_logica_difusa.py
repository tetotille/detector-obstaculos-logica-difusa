import cv2
import numpy as np

from object_detector import ObjectDetector
from utils import aplicar_logica_difusa_posicion,aplicar_logica_difusa_pixeles

class ColorDetector(ObjectDetector):
    """
    A class that detects objects of a specific color range in a given frame using the HSV color space.

    Parameters:
    frame (numpy.ndarray): The input frame.
    **kwargs: Optional keyword arguments.
        - rango_hsv (list): The lower and upper bounds of the color range in HSV format. Default is [[0, 100, 0], [100, 255, 200]].
        - scale (int): The percentage scale of the input frame. Default is 40.

    Attributes:
    anormal_pixels (int): The number of pixels that fall outside the specified color range.
    rango_hsv (list): The lower and upper bounds of the color range in HSV format.
    scale (int): The percentage scale of the input frame.
    width (int): The width of the resized frame.
    height (int): The height of the resized frame.
    frame (numpy.ndarray): The resized input frame.

    Example usage:
    >>> frame = cv2.imread("image.jpg")
    >>> detector = ColorDetector(frame, rango_hsv=[[30, 150, 50], [255, 255, 180]], scale=50)
    """
    def __init__(self,frame:np.ndarray,**kwargs):
        super().__init__(frame)
        self.anormal_pixels = 0
        self.rango_hsv = kwargs.get("rango_hsv",[[0, 100, 0], [100, 255, 200]])
        self.scale = kwargs.get("scale",40)

        self.width = int(frame.shape[1] * self.scale / 100)
        self.height = int(frame.shape[0] * self.scale / 100)
        dim = (self.width, self.height)
        self.frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    
    async def detect(self):
        """
        An asynchronous method that detects objects of a specific color range in the input frame and calculates their position and size.

        Parameters:
        self (ColorDetector): The instance of the class that this method belongs to.

        Returns:
        None

        Example usage:
        >>> frame = cv2.imread("image.jpg")
        >>> detector = ColorDetector(frame, rango_hsv=[[30, 150, 50], [255, 255, 180]], scale=50)
        >>> await detector.detect()
        """
        result,mask = await self.__color_detector()
        self.centroid = await self.__calculate_centroid(mask)
        self.position, self.left_grade, self.center_grade, self.right_grade = await aplicar_logica_difusa_posicion(self.centroid, (self.width, self.height), mostrar_grafica=False)
        pixels_num = cv2.countNonZero(mask)
        self.anormal_pixels = pixels_num
        total_pixels = self.width * self.height
        self.object_detected = aplicar_logica_difusa_pixeles(pixels_num, total_pixels, mostrar_grafica=False)

    def get_pixels(self):
        """
        A method that returns the number of pixels that fall outside the specified color range in the input frame.

        Parameters:
        self (ColorDetector): The instance of the class that this method belongs to.

        Returns:
        anormal_pixels (int): The number of pixels that fall outside the specified color range.

        Example usage:
        >>> frame = cv2.imread("image.jpg")
        >>> detector = ColorDetector(frame, rango_hsv=[[30, 150, 50], [255, 255, 180]], scale=50)
        >>> await detector.detect()
        >>> pixels = detector.get_pixels()
        """
        return self.anormal_pixels

    async def __calculate_centroid(self,mask):
        """
        Private method that calculates the centroid of a binary mask.

        Parameters:
        self (MyDetector): The instance of the class that this method belongs to.
        mask (numpy.ndarray): A binary mask.

        Returns:
        centroid (tuple): The centroid of the binary mask as a tuple of (x, y) coordinates.

        Example usage:
        >>> mask = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)
        >>> detector = ColorDetector(None, None)
        >>> centroid = await detector._MyDetector__calculate_centroid(mask)
        """
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        return (cX, cY)

    async def __color_detector(self):
        """
        Private method that detects a specific color range in a given frame using the HSV color space.

        Parameters:
        self (MyDetector): The instance of the class that this method belongs to.

        Returns:
        result (numpy.ndarray): A frame with only the pixels that fall within the specified color range.
        mask (numpy.ndarray): A binary mask that indicates which pixels fall within the specified color range.

        Example usage:
        >>> frame = cv2.imread("image.jpg")
        >>> detector = MyDetector(frame, [[30, 150, 50], [255, 255, 180]])
        >>> result, mask = detector._MyDetector__color_detector()
        """
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        low_range = np.array(self.rango_hsv[0])
        high_range = np.array(self.rango_hsv[1])
        mask = cv2.inRange(hsv, low_range, high_range)
        result = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        return result, mask
    
if __name__ == "__main__":
<<<<<<< HEAD
    image_path = json.load(open("config.json"))["img_path"] + "amanecer1.jpeg"
    imagen_camara = cv2.imread(image_path)
    scale_percent = 40
    width = int(imagen_camara.shape[1] * scale_percent / 100)
    height = int(imagen_camara.shape[0] * scale_percent / 100)
    dim = (width, height)
    imagen_camara = cv2.resize(imagen_camara, dim, interpolation=cv2.INTER_AREA)
=======
    import asyncio
    import os
    from sys import argv
    from os.path import abspath,dirname,join
>>>>>>> d34f8827d340a8c0e72db58e9b84fd6fb160eb1f

    async def main():
        if len(argv) <= 1: raise(NameError("You have to enter a file name as an argument"))
        if os.path.dirname(argv[1]):
            image = cv2.imread(argv[1])
        else:
            img_path = join(dirname(dirname(abspath(__file__))),f"img/{argv[1]}")
            image = cv2.imread(img_path)


        detector = ColorDetector(image)
        await detector.detect()
        
        print("Centroide:", detector.get_centroid())
        print("Posición:", detector.get_position())
        membership = detector.get_membership()
        print(f"Grado de pertenencia a la izquierda: {membership['left']}")
        print(f"Grado de pertenencia en el centro: {membership['center']}")
        print(f"Grado de pertenencia a la derecha: {membership['right']}")

        print("Número de píxeles:", detector.get_pixels())

        print("¿Es un objeto?:", detector.get_detection())
    
    asyncio.run(main())