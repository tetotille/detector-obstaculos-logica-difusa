import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import time

# Añadir WaSR-T al path
project_root = Path(__file__).resolve().parent.parent
wasr_t_path = project_root / "WaSR-T"
sys.path.append(str(wasr_t_path))

from wasr_t.data.transforms import PytorchHubNormalization
from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.utils import load_weights
from src.utils.utils import mask_to_bounding_boxes, draw_bounding_boxes

# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

def process_video_wasrt(video_path, weights_path):
    print(f"Iniciando procesamiento WaSR-T en CPU: {video_path}")
    
    # Configuración del modelo
    device = torch.device('cpu')
    model = wasr_temporal_resnet101(pretrained=False, hist_len=5)
    state_dict = load_weights(weights_path)
    model.load_state_dict(state_dict)
    model = model.sequential() # Modo secuencial (stateful)
    model = model.eval().to(device)
    model.clear_state()

    # Normalización
    transform = PytorchHubNormalization()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Redimensionamos para que WaSR-T no sea extremadamente lento en CPU
    target_size = (512, 384) 
    
    output_path = "main_output/tesis_wasrt_cpu.mp4"
    os.makedirs("main_output", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    display_available = "DISPLAY" in os.environ
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()

        # Preprocesamiento
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_resized = img_pil.resize(target_size, Image.BILINEAR)
        # PytorchHubNormalization espera una imagen PIL o numpy array si contiene ToTensor()
        img_normalized = transform(img_resized)
        
        # Batch de tamaño 1
        input_batch = {'image': img_normalized.unsqueeze(0).to(device)}

        # Inferencia
        with torch.no_grad():
            res = model(input_batch)
        
        # Postprocesamiento
        probs = res['out'].cpu().detach().numpy()
        pred_mask_idx = probs.argmax(1)[0].astype(np.uint8)
        pred_mask_color = SEGMENTATION_COLORS[pred_mask_idx]
        
        # Convertir a BGR para OpenCV
        res_bgr = cv2.cvtColor(pred_mask_color, cv2.COLOR_RGB2BGR)
        
        # Superponer sobre la imagen original (redimensionada)
        img_original_resized = cv2.resize(frame, target_size)
        # Asegurarse de que res_bgr tenga el mismo tamaño exacto
        if res_bgr.shape[1] != target_size[0] or res_bgr.shape[0] != target_size[1]:
            res_bgr = cv2.resize(res_bgr, target_size)
            
        blended = cv2.addWeighted(img_original_resized, 0.5, res_bgr, 0.5, 0)

        # Algoritmo de encuadre sencillo para obstáculos (clase 0 en WaSR-T)
        obs_mask = (pred_mask_idx == 0).astype(np.uint8)
        if obs_mask.shape[1] != target_size[0] or obs_mask.shape[0] != target_size[1]:
            obs_mask = cv2.resize(obs_mask, target_size, interpolation=cv2.INTER_NEAREST)
            
        cuadros_wasrt = mask_to_bounding_boxes(obs_mask, min_area=30)
        for c in cuadros_wasrt:
            cv2.rectangle(blended, (c["x_init"], c["y_init"]), (c["x_end"], c["y_end"]), (0, 0, 255), 2)
            cv2.putText(blended, f"WaSR-T: {c['weight']}px", (c["x_init"], max(15, c["y_init"] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #     except:
        #         pass
        
        out.write(blended)

        if frame_count % 5 == 0:
            print(f"WaSR-T Frame {frame_count} procesado. Tiempo: {time.time() - start_time:.3f}s")
            
    cap.release()
    out.release()
    if display_available:
        cv2.destroyAllWindows()
    print(f"Procesamiento WaSR-T finalizado. Video guardado en: {output_path}")

if __name__ == "__main__":
    video_file = "/home/tetotille/Proyectos/detector-obstaculos-logica-difusa/assets/videos/tesis.mp4"
    weights_file = "/home/tetotille/Proyectos/detector-obstaculos-logica-difusa/WaSR-T/wasrt_mastr1478.pth"
    process_video_wasrt(video_file, weights_file)
