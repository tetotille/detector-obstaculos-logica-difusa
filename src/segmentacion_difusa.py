from os.path import dirname, abspath, join
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# Función para cargar y preprocesar la imagen
def load_and_preprocess_image(image_path):
    image = io.imread(image_path)
    grayscale_image = color.rgb2gray(image)
    return grayscale_image

# Función para inicializar los centroides
def initialize_centroids(X, n_clusters):
    np.random.seed(42)
    centroids = [X[np.random.randint(0, len(X))]]
    for _ in range(1, n_clusters):
        dist_sq = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in X])
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        i = np.searchsorted(cumulative_probs, r)
        centroids.append(X[i])
    return np.array(centroids)

# Función para calcular la distancia
def calculate_distance(x, c):
    return np.abs(x - c)

# Función para calcular la matriz de membresía
def update_membership_matrix(X, centroids, m):
    N = len(X)
    K = len(centroids)
    U = np.zeros((N, K))

    for i in range(N):
        for k in range(K):
            denom_sum = sum([(calculate_distance(X[i], centroids[k]) / calculate_distance(X[i], centroids[j]))**(2/(m-1)) for j in range(K)])
            U[i, k] = 1 / denom_sum

    return U

# Implementación del algoritmo Fuzzy C-Means
def fuzzy_c_means(image, n_clusters, m, max_iter=100, error=1):
    N = image.size
    X = image.flatten()
    
    # Normalizar los datos
    X = (X - X.min()) / (X.max() - X.min())
    print(X)
    # Inicializar centroides y la matriz de membresía
    centroids = initialize_centroids(X, n_clusters)
    U = np.random.dirichlet(np.ones(n_clusters), size=N)
    
    for iteration in range(max_iter):
        U_old = U.copy()
        
        # Actualizar los centroides
        for k in range(n_clusters):
            num = np.sum((U[:, k]**m) * X)
            denom = np.sum(U[:, k]**m)
            centroids[k] = num / denom
            print(f'Iteración: {iteration}, Centroid {k}: {centroids[k]}')
        
        # Actualizar la matriz de membresía
        U = update_membership_matrix(X, centroids, m)
        
        # Verificar la convergencia
        diff = np.linalg.norm(U - U_old)
        print(f'Iteración: {iteration}, Diferencia: {diff}')
        if diff < error:
            print(f'Convergencia alcanzada en la iteración {iteration}')
            break
    
    if iteration == max_iter - 1:
        print('No se alcanzó la convergencia en el número máximo de iteraciones')
    
    return U, centroids

# Función para interpretar los grados de membresía
def interpret_membership(U):
    high_threshold = 0.7
    low_threshold = 0.3
    membership_degrees = np.zeros_like(U)
    
    membership_degrees[U >= high_threshold] = 2  # Alto grado
    membership_degrees[(U < high_threshold) & (U >= low_threshold)] = 1  # Medio grado
    membership_degrees[U < low_threshold] = 0  # Bajo grado
    
    return membership_degrees

# Función para crear la imagen segmentada
def create_segmented_image(image, U):
    segmented_image = np.zeros_like(image)
    membership_degrees = interpret_membership(U)
    cluster_assignments = np.argmax(U, axis=1)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_index = i * image.shape[1] + j
            segmented_image[i, j] = cluster_assignments[pixel_index]
    
    return segmented_image, membership_degrees

# Parámetros del algoritmo
max_iter = 1000
error = 1e-4
    # Ruta a la imagen
if len(argv) > 1:
    filename = join(dirname(dirname(abspath(__file__))), f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))), "img/barco.jpg")

n_clusters = 10
m = 5
image_path = filename
# Ejecutar el algoritmo
image = load_and_preprocess_image(image_path)
U, centroids = fuzzy_c_means(image, n_clusters, m)
segmented_image, membership_degrees = create_segmented_image(image, U)

# Mostrar la imagen original y la segmentada
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Imagen Original")
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Imagen Segmentada")
plt.imshow(segmented_image, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Grados de Membresía")
plt.imshow(membership_degrees.max(axis=1).reshape(image.shape), cmap='viridis')
plt.colorbar(ticks=[0, 1, 2], label='Grado de Membresía')
plt.show()
