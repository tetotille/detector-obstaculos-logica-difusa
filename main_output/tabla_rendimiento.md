# Comparación Cuantitativa y de Rendimiento en NVIDIA Jetson Orin Nano

## 1. Detección de Obstáculos en Video Real (Evaluación de Presencia por Frame)

Evaluación efectuada sobre **181 frames** del video `assets/videos/tesis.mp4` (paso de 10 cuadros).
- **Región de Interés (ROI):** Superficie de agua navegable delimitada verticalmente debajo de la línea estimada del horizonte ($y \ge y_{\text{horizonte}}$) hasta el borde inferior.
- **Regla de presencia:** Se registra detección positiva si existe $\ge 1$ caja delimitadora dentro de la ROI con área $\ge 30\text{ px}$ y activación de obstáculo.

| Método | Frames | TP | FP | FN | TN | Precisión | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaSR-T (Secuencial stateful)** | 181 | 124 | 3 | 28 | 26 | **0.976** | **0.816** | **0.889** |
| **Ours (Difuso sin memoria)** | 181 | 59 | 9 | 93 | 20 | **0.868** | 0.388 | 0.536 |

### Verificación de Consistencia Metodológica:
- **Total frames evaluados:** 181 en ambos métodos.
- **TP + FN (Total Positivos Reales GT):** Ours = 152 | WaSR-T = 152 (**Coincidencia exacta**).
- **FP + TN (Total Negativos Reales GT):** Ours = 29 | WaSR-T = 29 (**Coincidencia exacta**).

---

## 2. Benchmark de Velocidad en Plataforma Única (NVIDIA Jetson Orin Nano 8GB)

Medición sobre los mismos cuadros del video `tesis.mp4`, excluyendo calentamiento (*warm-up*) y llamadas de dibujo/renderizado gráfico. Sincronización CUDA garantizada con `torch.cuda.synchronize()`.

- **Plataforma:** NVIDIA Jetson Orin Nano Developer Kit (8GB RAM unificada)
- **Modo de Potencia:** `NV Power Mode: 25W` (Modo 1 de máximo rendimiento)

| Método | Backend | Resolución | Tiempo Medio por Frame | Throughput (FPS) | Estado para Bucle de Control |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Ours (Fuzzy Logic)** | **GPU (PyTorch CUDA)** | **256×192** | **24.86 ± 2.0 ms** | **40.23 FPS** | **Tiempo real pleno (>30 FPS)** |
| **Ours (Fuzzy Logic)** | **CPU (ARM Cortex-A78AE)** | **256×192** | **188.10 ± 8.8 ms** | **5.32 FPS** | Operable a baja velocidad |
| **WaSR-T** | GPU (NVIDIA Ampere CUDA) | 512×384 | 841.99 ± 18.4 ms | 1.19 FPS | Marginal (retardo de ~0.84 s) |
| **WaSR-T** | CPU (ARM Cortex-A78AE) | 512×384 | 16721.22 ± 222.3 ms | 0.06 FPS | Inviable en CPU embebido |

### Aspectos Destacados de la Implementación en GPU:
1. **Aceleración GPU del Algoritmo Difuso:** Al migrar el cálculo matricial y las distancias de Fuzzy C-Means (FCM) a tensores de GPU mediante PyTorch CUDA, el tiempo por frame del algoritmo difuso desciende de **188.10 ms** a **24.86 ms** (**aceleración de 7.5×**).
2. **Ventaja frente a Redes Neuronales:** La versión difusa en GPU es **33.8× más rápida que WaSR-T en la misma GPU** (40.23 FPS vs 1.19 FPS), garantizando un margen de procesamiento holgado para algoritmos de planificación de rutas (A*, RRT*) y odometría visual/LiDAR en el vehículo autónomo.
