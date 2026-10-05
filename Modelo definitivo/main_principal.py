import cv2
import os
from os.path import join, dirname, abspath
from cp_contorno_difuso import process_image
import numpy as np

def main():
    print("Script iniciado (Adaptado para CPU)")
    
    # Path de la imagen
    base_path = dirname(dirname(abspath(__file__)))
    filename = join(base_path, "assets/images/barco.jpg")
    
    # Fallback al path absoluto si es necesario
    if not os.path.exists(filename):
        filename = "/home/tetotille/Proyectos/detector-obstaculos-logica-difusa/assets/images/barco.jpg"
    
    print(f"Buscando imagen en: {filename}")

    if not os.path.exists(filename):
        print(f"Error: No se encontró la imagen en {filename}")
        return

    # Cargar la imagen
    image = cv2.imread(filename)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {filename}")
        return
        
    print("Imagen cargada con éxito. Procesando en CPU...")

    # Procesar imagen
    processed_image = process_image(image)
    
    # Guardar resultado (en lugar de cv2.imshow para evitar problemas sin display)
    output_name = "imagen_umbral.png"
    cv2.imwrite(output_name, processed_image)
    print(f"Procesamiento completado. Imagen guardada como: {output_name}")

if __name__ == "__main__":
    main()
