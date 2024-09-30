import cv2
from os.path import join, dirname, abspath
from cp_contorno_difuso import process_image
import cupy as cp

def main():
    print("Script iniciado")
    filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")
    print(filename)  # Esto te permitirá verificar la ruta completa

    # Cargar y mostrar la imagen original
    image = cv2.imread(filename)
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    # Llamar a la función de procesamiento y obtener la imagen procesada
    processed_image = process_image(image)
        # Convertir la imagen de CuPy a NumPy
    processed_image_np = cp.asnumpy(processed_image)
    # Mostrar la imagen procesada
    cv2.imshow('Imagen Umbral', processed_image_np)
    cv2.waitKey(0)

    # Guardar la imagen procesada
    cv2.imwrite("imagen_umbral.png", processed_image_np)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
