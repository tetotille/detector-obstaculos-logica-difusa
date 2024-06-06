# Teoría de Lógica Difusa Implementada en el Código
    La lógica difusa es un enfoque matemático para manejar la incertidumbre y la imprecisión, que es particularmente útil en sistemas de control y toma de decisiones. A diferencia de la lógica clásica que opera con valores binarios (verdadero/falso), la lógica difusa trabaja con grados de verdad que pueden variar entre 0 y 1. En este código, se implementa la lógica difusa para dos propósitos principales: determinar la posición de un objeto en una imagen y evaluar la significancia de la cantidad de píxeles detectados.
1. Dominios de Discurso y Funciones de Pertenencia
    Un dominio de discurso es el conjunto de todos los posibles valores que una variable difusa puede tomar. En este código, hemos definido dos dominios: uno para la posición en el eje X (horizontal) y otro para la posición en el eje Y (vertical).
Posición Horizontal (X):
- Izquierda: Pertenencia a la parte izquierda de la imagen.
- Centro: Pertenencia a la parte central de la imagen.
- Derecha: Pertenencia a la parte derecha de la imagen.
Posición Vertical (Y):
- Arriba: Pertenencia a la parte superior de la imagen.
- Abajo: Pertenencia a la parte inferior de la imagen.
Cada una de estas categorías se representa mediante funciones de pertenencia, que pueden ser de diferentes formas, como triangulares o sigmoides. Las funciones de pertenencia asignan un grado de pertenencia (un valor entre 0 y 1) a cada punto en el dominio.
2. Funciones de Pertenencia Triangulares y Sigmoides
    Funciones de Pertenencia Triangulares: Estas funciones tienen la forma de un triángulo y se utilizan para determinar la pertenencia de un valor a una categoría en función de su proximidad a un punto medio. Ejemplo:
```python
    pos_x.left = triangular(0, ancho/2)
    pos_x.center = triangular(ancho/4, (3*ancho)/4)
    pos_x.right = triangular(ancho/2,  ancho)
```
    Funciones de Pertenencia Sigmoides: Estas funciones tienen la forma de una S y se utilizan para suavizar las transiciones entre categorías. Ejemplo:
```python
    pixeles.pocos = bounded_sigmoid(0, muchos_umbral, inverse=True)
    pixeles.muchos = bounded_sigmoid(pocos_umbral, total_pixeles, inverse=True)
```
3. Evaluación de Grados de Pertenencia
    Para determinar la pertenencia de un valor (como una coordenada o el número de píxeles) a una categoría, se calcula su grado de pertenencia utilizando la función de pertenencia correspondiente. Ejemplo:
```python
    grado_izq = pos_x.left(x)
    grado_centro = pos_x.center(x)
    grado_der = pos_x.right(x)
```
4. Toma de Decisiones Difusas
La toma de decisiones difusas se basa en los grados de pertenencia calculados. Se selecciona la categoría con el mayor grado de pertenencia como el resultado final. Ejemplo:
```python
    if grado_izq >= grado_centro and grado_izq >= grado_der:
        pos_x_res = "a la izquierda"
    elif grado_centro >= grado_izq and grado_centro >= grado_der:
        pos_x_res = "en el centro"
    else:
        pos_x_res = "a la derecha"
```
5. Lógica Difusa para la Cantidad de Píxeles
    Para evaluar si la cantidad de píxeles detectados representa un objeto significativo, se define un dominio de discurso basado en el número total de píxeles de la imagen. Las funciones de pertenencia sigmoides se utilizan para determinar si el número de píxeles es "poco" o "mucho".
### Cálculo de Umbrales:
```python
    pocos_umbral = 0.15 * total_pixeles
    muchos_umbral = 0.10 * total_pixeles
```
### Evaluación de Grados de Pertenencia:
```python
    grado_pocos = pixeles.pocos(num_pixeles)
    grado_muchos = pixeles.muchos(num_pixeles)
```
### Decisión:
```python
if grado_muchos > grado_pocos:
    return "Es un objeto"
else:
    return "No es un objeto"
```
## Conclusión
    La implementación de la lógica difusa en este código permite una evaluación flexible y precisa de la posición y significancia de objetos en una imagen. Utilizando funciones de pertenencia triangulares y sigmoides, el sistema puede manejar la incertidumbre y proporcionar resultados que se ajustan dinámicamente a las características de la imagen. Este enfoque es útil en aplicaciones donde la precisión y la adaptabilidad son cruciales, como en la detección y seguimiento de objetos en tiempo real.