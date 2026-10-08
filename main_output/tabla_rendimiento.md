# Comparación Cuantitativa y de Rendimiento en NVIDIA Jetson Orin Nano

## 1. Rendimiento y Latencia en Hardware Embebido (Jetson Orin Nano @ 25W)

| Método | Resolución | Precision | Recall | F1-Score | Latencia (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaSR-T (Temporal CNN)** | 512×384 | **96.08%** | **69.50%** | **80.66%** | 831.0 ms | 1.20 FPS |
| **Ours (Fuzzy Logic)** | 256×192 | 83.64% | 32.62% | 46.94% | **43.7 ms** | **22.86 FPS** |

---

## 2. Matriz de Confusión Unificada (180 Cuadros Evaluados)

Ground Truth unificado: **141 positivos reales** y **39 negativos reales** (total = 180).

| Método | TP | FP | FN | TN | Total | Precisión | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaSR-T (Temporal CNN)** | 98 | 4 | 43 | 35 | 180 | 96.08% | 69.50% | 80.66% |
| **Ours (Fuzzy Logic)** | 46 | 9 | 95 | 30 | 180 | 83.64% | 32.62% | 46.94% |

### Verificación Matemática:
* **Positivos Reales:** $TP_{ours} + FN_{ours} = 46 + 95 = 141$ | $TP_{wasrt} + FN_{wasrt} = 98 + 43 = 141$
* **Negativos Reales:** $FP_{ours} + TN_{ours} = 9 + 30 = 39$ | $FP_{wasrt} + TN_{wasrt} = 4 + 35 = 39$
* **Total de Cuadros:** $180$ para ambos métodos.
