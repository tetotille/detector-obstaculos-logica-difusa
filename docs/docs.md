# Documentación

## Detección de Línea del horizonte

### Horizonte mar rojo cielo azul

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

#### Status

Aun no funciona del todo

#### Algoritmo

- Verifica qué tanta rotación tiene la imagen

- Debe estar inclinado más de 10deg

- Según el resultado utiliza imagen inclinada o horizonte mar rojo cielo azul

---

## Detectores de Objetos

### Detección de Color

#### Status

Funciona súper bien.

#### Algoritmo

- Convierte a HSV.

- Verifica las máscaras de colores en un rango que no es común para el paisaje.

- Saltan los objetos que tienen mucho color distinto al paisaje.

---

### Detección Objeto Mov

#### Status

Funciona a medias

#### Algoritmo

- Para objetos cercanos funciona mejor

- Compara dos frames en un video y encuentra las diferencias del mismo.

- Las diferencias las encierra en un rectángulo.

---

### Detección Objeto Tex

#### Status

Funciona

#### Algoritmo

- Utiliza los contornos de la imagen.

- Al encontrar contornos referenciales los encierra en rectángulos.
