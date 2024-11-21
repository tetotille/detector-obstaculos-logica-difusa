import cv2

from os.path import dirname, abspath,join

#from src.cmeans import fcm
from src.cmeans import c_means_main
#from src.detector_horizonte import find_largest_fuzzy_jump, separate_pixels
from src.detector_horizonte import pixel_detector
#from src.detector_hsv import detector_hsv,detector_rgb
from src.detector_hsv import detector_hsv,detector_rgb
from src.utils import read_image



image_folder = join(dirname(dirname(abspath(__file__))),"assets/images")
image_set = {f"{join(image_folder,'akaso1.jpeg')}",
             f"{join(image_folder,'akaso2.jpeg')}",
            #  f"{join(image_folder,'ypacarai.jpeg')}",
            #  f"{join(image_folder,'barco.jpg')}",
            #  f"{join(image_folder,'lago-ypacarai (6).jpg')}",
            #  f"{join(image_folder,'IMG_6830.jpeg')}",
             f"{join(image_folder,'akaso3.jpeg')}",}

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
    # print(f"Image name: {image_path.split('/')[-1]}")
    image = cv2.imread(image_path)
    filename = image_path.split('/')[-1]
    return image,filename

def main():
    # Cuando se use el video
    # while True:
        # frame = get_video_stream()

    # Cuando se use una imagen
    import pickle
    i = 0
    tics = {}
    repetir = 1
    while image_set:
        i += 1
        frame,filename = get_image_frame()
        tics[filename] = {"total":[],"horizon":[],"rgb":[],"cmeans":[]}
        for i in range(repetir):
            init = time()
            image_np,image_cp = read_image(frame,256,192)
            left,center,right = separate_pixels(image_np)
            
            left = find_largest_fuzzy_jump(left)
            center = find_largest_fuzzy_jump(center)
            right = find_largest_fuzzy_jump(right)

            tic1 = time()
            if abs(center - left) < abs(right - center) and abs(center - left) < abs(right - left):
                a,b = left,center
            elif abs(center - left) > abs(right - center) and abs(right - center) < abs(right - left):
                a,b = center,right
            else:
                a,b = left,right
            
            # Draw horizontal lines at the fuzzy jump indices
            #cv2.line(image_np, (0, a), (image_np.shape[1], b), (0, 255, 0), 2)
            cropped_image_np = image_np[(a+b)//2:,:]
            cropped_image_cp = image_cp[(a+b)//2:,:]

            ajuste = (a+b)//2
            tic2 = time()
            hsv_np, cuadros_rgb = detector_rgb(cropped_image_np)
            tic3 = time()
            cmeans_image, cmeans_fila_interes, cuadros_cmeans = fcm(image_np,4)
            tic4 = time()
            encuadrar = True
            if encuadrar:
                for cuadro in cuadros_rgb:
                    cv2.rectangle(image_np, (cuadro["x_init"],cuadro["y_init"]+ajuste), (cuadro["x_end"],cuadro["y_end"]+ajuste), (0, 0, 255), 2)
                for cuadro in cuadros_cmeans:
                    cv2.rectangle(image_np, (cuadro["x_init"],cuadro["y_init"]+cmeans_fila_interes), (cuadro["x_end"],cuadro["y_end"]+cmeans_fila_interes), (0, 255, 0), 2)

                cv2.imwrite("main_output/" + filename.split(".")[0] + ".png", image_np)
                cv2.imwrite("main_output/rgb_" + filename.split(".")[0] + ".png", hsv_np)
                cv2.imwrite("main_output/cmeans_" + filename.split(".")[0] + ".png", cmeans_image)
            
            tics[filename]["total"].append(time()-init)
            tics[filename]["horizon"].append(tic1-init)
            tics[filename]["rgb"].append(tic3-tic2)
            tics[filename]["cmeans"].append(tic4-tic3)
            
            # print("RGB Detector")
            # print(cuadros_rgb)
            # print("\n\nCmeans Detector")
            # print(cuadros_cmeans)
            print(f"Processing\t{i}/100",end="\r")
    pickle.dump(tics,open("main_output/tics.pkl","wb"))


if __name__ == "__main__":

    from time import time

    main()