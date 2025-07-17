import cupy as cp
import cv2

def proccess_image(image:cp.array):
    height, width = image.shape[:2]
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    gpu_kernel = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, image.dtype, kernel)
    gpu_dilated = gpu_kernel.apply(gpu_frame)

    dilated_image = gpu_dilated.download()