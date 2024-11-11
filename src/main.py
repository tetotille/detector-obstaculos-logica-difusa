import cv2

from os.path import dirname, abspath,join

from src.detector_horizonte import find_largest_fuzzy_jump, separate_pixels
from src.utils import read_image


image_folder = join(dirname(dirname(abspath(__file__))),"assets/images")
image_set = {f"{join(image_folder,'akaso1.jpeg')}",f"{join(image_folder,'akaso2.jpeg')}",f"{join(image_folder,'akaso3.jpeg')}"}

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
    while image_set:
        frame = get_image_frame()
        image_np,image_cp = read_image(frame,256,192)
        left,center,right = separate_pixels(image_np)
        
        left_max_fuzzy_jump_index = find_largest_fuzzy_jump(left)
        center_max_fuzzy_jump_index = find_largest_fuzzy_jump(center)
        right_max_fuzzy_jump_index = find_largest_fuzzy_jump(right)

        print(f"left_max_fuzzy_jump_index: {left_max_fuzzy_jump_index}")
        print(f"center_max_fuzzy_jump_index: {center_max_fuzzy_jump_index}")
        print(f"right_max_fuzzy_jump_index: {right_max_fuzzy_jump_index}")





if __name__ == "__main__":
    main()