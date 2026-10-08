import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
DIR_PRUEBAS = project_root / "main_output/pruebas_visuales"

def main():
    images = sorted(list(DIR_PRUEBAS.glob("*.png")))
    if not images:
        print("[!] No se encontraron imágenes en:", DIR_PRUEBAS)
        return

    win_name = "Visor de Pruebas: Segmentacion FCM Previa + Encuadre Posterior"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1400, 380)

    idx = 0
    print("\n[INSTRUCCIONES DEL VISOR]")
    print("  [d] / [ESPACIO] / [Flecha Der] : Siguiente imagen")
    print("  [a] / [Flecha Izq]            : Imagen anterior")
    print("  [q] / [ESC]                   : Salir")
    print(f"  Total imagenes disponibles: {len(images)}\n")

    while True:
        img_path = images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            break

        cv2.imshow(win_name, img)
        key = cv2.waitKey(0) & 0xFF

        if key in [ord('q'), 27]:
            break
        elif key in [ord('d'), ord(' '), 83]:
            idx = (idx + 1) % len(images)
        elif key in [ord('a'), 81]:
            idx = (idx - 1) % len(images)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
