# Comparación Cuantitativa y de Rendimiento en NVIDIA Jetson Orin Nano

## 1. Rendimiento y Latencia en Hardware Embebido (Jetson Orin Nano @ 25W)

| Método | Resolución | Precision | Recall | F1-Score | Latencia (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaSR-T (Temporal CNN)** | 512×384 | **91.96%** | **74.10%** | **82.07%** | 831.0 ms | 1.20 FPS |
| **Ours (Fuzzy Logic)** | 256×192 | 92.59% | 53.96% | 68.18% | **43.7 ms** | **22.86 FPS** |

---

## 2. Matriz de Confusión Unificada (180 Cuadros Evaluados)

Ground Truth unificado: **139 positivos reales** y **41 negativos reales** (total = 180).

| Método | TP | FP | FN | TN | Total | Precisión | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaSR-T (Temporal CNN)** | 103 | 9 | 36 | 32 | 180 | 91.96% | 74.10% | 82.07% |
| **Ours (Fuzzy Logic)** | 75 | 6 | 64 | 35 | 180 | 92.59% | 53.96% | 68.18% |

### Verificación Matemática:
* **Positivos Reales:** $TP_{ours} + FN_{ours} = 75 + 64 = 139$ | $TP_{wasrt} + FN_{wasrt} = 103 + 36 = 139$
* **Negativos Reales:** $FP_{ours} + TN_{ours} = 6 + 35 = 41$ | $FP_{wasrt} + TN_{wasrt} = 9 + 32 = 41$
* **Total de Cuadros:** $180$ para ambos métodos.
