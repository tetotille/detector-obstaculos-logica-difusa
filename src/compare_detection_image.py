"""
Entry point for compare_detection_image (English alias of comparar_deteccion_imagen.py).
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.comparar_deteccion_imagen import compare_detection, comparar_deteccion, main

if __name__ == "__main__":
    main()
