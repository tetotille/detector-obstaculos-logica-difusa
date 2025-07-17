# Documentación

## Detección de Línea del horizonte

### Horizonte mar rojo cielo azul

#### Referencias

- https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
- https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
- https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
- https://docs.python.org/3/library/json.html
  
  #### Status

Funcionó para todas las imágenes menos para la inclinada

#### Algoritmo

- Se halla el gradiente de color.

- Se utliza la máscara del cielo que va desde arriba hasta la línea del horizonte.

- Desde abajo hasta arriba te barre el agua que interpreta como otra máscara.

- Donde se intersectan las máscaras es la línea del horizonte.

```python
def detectar_horizonte(image):
    """ Halla la línea del horizonte a través de las colisiones de dos gradientes de color.

    ### Args:
        image (Image): Imagen de el agua

    ### Returns:
        image, transition_index: retorna la imagen dibujada con la línea del horizonte y la línea completa del horizonte
    """

    # Filtros normales
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = np.gradient(blurred_image, axis=0)
    smoothed_gradient = np.convolve(np.mean(gradient, axis=1), np.ones(15)/15, mode='same')

    # Encontrar el punto de transición más significativo y dibujar la línea de horizonte
    transition_index = np.argmax(np.abs(smoothed_gradient))
    if transition_index is not None:
        cv2.line(image, (0, transition_index), (image.shape[1], transition_index), (0, 255, 0), thickness=2)

    return image, transition_index
```

---

### Imagen Inclinada

#### Referencias

- https://docs.opencv.org/4.x/
- https://numpy.org/doc/
- https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
- https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
- https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
- https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/
- https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
- https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
- https://www.geeksforgeeks.org/image-resizing-using-opencv-python/

#### Status

Funcionó para alguna que otra imagen, no es muy preciso.

#### Algoritmo

- Aplica filtro Canny

- Busca la línea más larga

- Obtiene los puntos extremos de la línea

- Con los puntos extremos grafica la línea inclinada

---

## Nexos de Funciones

### Rotación Imagen

#### Referencias

- https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
- https://omes-va.com/operadores-bitwise/
- https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
- https://numpy.org/doc/
- https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
- https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/

#### Status

Aun no funciona del todo

#### Algoritmo

- Verifica qué tanta rotación tiene la imagen

- Debe estar inclinado más de 10deg

- Según el resultado utiliza imagen inclinada o horizonte mar rojo cielo azul

---

## Detectores de Objetos

### Detección de Color

#### Referencias

- https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
- https://omes-va.com/operadores-bitwise/
- https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
- https://numpy.org/doc/
- https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
- https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/

#### Status

Funciona súper bien.

#### Algoritmo

- Convierte a HSV.

- Verifica las máscaras de colores en un rango que no es común para el paisaje.

- Saltan los objetos que tienen mucho color distinto al paisaje.

---

### Detección Objeto Mov

#### Referencia

- https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
- https://stackoverflow.com/questions/71358885/correct-if-absdiff-the-same-image-just-slightly-shifted-vertically-the-diffe
- https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
- https://www.geeksforgeeks.org/python-opencv-waitkey-function/
- https://stackoverflow.com/questions/48213499/whats-the-meaning-of-cv2-videocapture-release
- https://www.geeksforgeeks.org/python-opencv-destroyallwindows-function/

#### Status

Funciona a medias

#### Algoritmo

- Para objetos cercanos funciona mejor

- Compara dos frames en un video y encuentra las diferencias del mismo.

- Las diferencias las encierra en un rectángulo.

---

### Detección Objeto Tex

#### Referencia

- https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/
- https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
- https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
- https://www.geeksforgeeks.org/python-opencv-distancetransform-function/
- https://www.tutorialspoint.com/how-to-normalize-an-image-in-opencv-python
- https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- https://numpy.org/doc/stable/user/basics.types.html
- https://www.geeksforgeeks.org/how-to-subtract-two-images-using-python-opencv/
- https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
- https://www.simplilearn.com/image-processing-article
- https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
  |
  
  #### Status

Funciona

#### Algoritmo

- Utiliza los contornos de la imagen.

- Al encontrar contornos referenciales los encierra en rectángulos.
