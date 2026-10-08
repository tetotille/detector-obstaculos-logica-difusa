# Comparación Cuantitativa y de Rendimiento en NVIDIA Jetson Orin Nano

## 1. Detección de Obstáculos en Video Real (Evaluación Cuadro a Cuadro)

Evaluación efectuada sobre **180 cuadros etiquetados** del video continuo `assets/videos/tesis.mp4` (paso de 10 cuadros, rango 0 a 1799) comparando el algoritmo propuesto renovado frente al modelo de referencia WaSR-T.

| Método | Total Cuadros | TP | FP | FN | TN | Precisión | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaSR-T (CNN ResNet-101 Temporal)** | 180 | 97 | 15 | 36 | 32 | **86.61%** | **72.93%** | **79.18%** |
| **Ours (Lógica Difusa Renovada)** | 180 | 58 | 23 | 67 | 32 | 71.60% | 46.40% | 56.31% |

### Análisis de Detección:
- **Verdaderos Negativos (Agua Limpia):** Ambos métodos logran 32 verdaderos negativos sobre el conjunto de cuadros de agua despejada, evidenciando que el umbral de contraste de 0.245 y la normalización de fondo rechazan eficazmente el oleaje en aguas limpias.
- **Precisión vs. Exhaustividad:** WaSR-T logra un mayor recall al detectar objetos lejanos de bajo contraste gracias a la memoria temporal y la profundidad de ResNet-101. El algoritmo propuesto prioriza el tiempo de respuesta instantáneo y el bajo costo computacional, manteniendo una precisión del 71.60%.

---

## 2. Benchmark de Eficiencia en Hardware Embebido (NVIDIA Jetson Orin Nano 8GB)

Medición sobre cuadros reales del video de navegación en el Lago Ypacaraí, con sincronización CUDA (`torch.cuda.synchronize()`), excluyendo cuadros de calentamiento (*warm-up*) y llamadas de renderizado de interfaz gráfica.

- **Plataforma:** NVIDIA Jetson Orin Nano Developer Kit (8GB RAM LPDDR5 unificada)
- **Modo de Potencia:** `NV Power Mode: 25W` (Máximo rendimiento)
- **IP:** 192.168.1.115

| Método | Backend | Resolución | Tiempo Medio por Frame | Throughput (FPS) | Memoria RAM Pico | Estado para Navegación USV |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Ours (Lógica Difusa)** | **GPU (PyTorch CUDA)** | **256×192** | **43.89 ± 1.1 ms** | **22.78 FPS** | **1544 MB** | **Tiempo real pleno (>20 FPS)** |
| **Ours (Lógica Difusa)** | CPU (ARM Cortex-A78AE) | 256×192 | 282.35 ± 15.4 ms | 3.54 FPS | 1052 MB | Operable a velocidad reducida |
| **WaSR-T** | GPU (NVIDIA Ampere CUDA) | 512×384 | 832.00 ± 13.5 ms | 1.20 FPS | 4242 MB | Marginal (latencia de 832 ms) |
| **WaSR-T** | CPU (ARM Cortex-A78AE) | 512×384 | 16419.06 ± 106.9 ms | 0.06 FPS | 3785 MB | Inviable en CPU embebida |

### Conclusiones de Rendimiento para el USV:
1. **Velocidad de Respuesta:** El algoritmo difuso en GPU corre a **22.78 FPS (43.89 ms)**, siendo **19.0 veces más rápido que WaSR-T** en la misma placa embebida, permitiendo una evasión ágil de obstáculos.
2. **Presupuesto de Memoria:** WaSR-T consume **4.24 GB (más del 53% de la memoria total)** de la Jetson Orin Nano, mientras que la solución difusa ocupa solo **1.54 GB**, dejando más de 6.4 GB libres para localización, LiDAR, SLAM y planificación de trayectorias.
