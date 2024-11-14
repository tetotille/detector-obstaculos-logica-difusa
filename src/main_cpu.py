import cv2

from os.path import dirname, abspath,join

from src.detector_horizonte import find_largest_fuzzy_jump, separate_pixels
from src.detector_hsv import detector_hsv,detector_rgb
from src.utils import read_image


image_folder = join(dirname(dirname(abspath(__file__))),"assets/images")
image_set = {f"{join(image_folder,'akaso1.jpeg')}",
             f"{join(image_folder,'akaso2.jpeg')}",
             f"{join(image_folder,'ypacarai.jpeg')}",
             f"{join(image_folder,'akaso3.jpeg')}",
             f"{join(image_folder,'barco.jpg')}",
             f"{join(image_folder,'lago-ypacarai (6).jpg')}",
             f"{join(image_folder,'IMG_6830.jpeg')}",}

def get_video_stream():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yield frame

    cap.release()

def get_image_frame():
    image_path = image_set.pop() 
    print(f"Image name: {image_path.split('/')[-1]}")
    image = cv2.imread(image_path)
    return image

def main():
    # Cuando se use el video
    # while True:
        # frame = get_video_stream()

    # Cuando se use una imagen
    i = 0
    while image_set:
        i += 1
        frame = get_image_frame()
        image_np,image_cp = read_image(frame,256,192)
        left,center,right = separate_pixels(image_np)
        
        left = find_largest_fuzzy_jump(left)
        center = find_largest_fuzzy_jump(center)
        right = find_largest_fuzzy_jump(right)


        if abs(center - left) < abs(right - center) and abs(center - left) < abs(right - left):
            a,b = left,center
        elif abs(center - left) > abs(right - center) and abs(right - center) < abs(right - left):
            a,b = center,right
        else:
            a,b = left,right
        
        # Draw horizontal lines at the fuzzy jump indices
        cv2.line(image_np, (0, a), (image_np.shape[1], b), (0, 255, 0), 2)
        image_np = image_np[(a+b)//2:,:]
        image_cp = image_cp[(a+b)//2:,:]

        hsv_np, cuadros = detector_rgb(image_np)
        encuadrar = True
        if encuadrar:
            for cuadro in cuadros:
                cv2.rectangle(image_np, (cuadro["x_init"],cuadro["y_init"]), (cuadro["x_end"],cuadro["y_end"]), (0, 0, 255), 2)

            cv2.imwrite(f"output_{i}.png", image_np)
        else:
            cv2.imwrite(f"output_{i}.png", hsv_np)
        



if __name__ == "__main__":
    main()