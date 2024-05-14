import cv2

class DetectorDeObstaculos:
    def __init__(self):
        self.centroide_anterior = None

    def detectar_obstaculo(self, frame):
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar umbralización para resaltar el obstáculo
        _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Encontrar contornos en la imagen umbralizada
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujar los contornos en el frame original
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filtrar contornos pequeños
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calcular centroide del contorno
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroide_actual = (cx, cy)

                    # Realizar seguimiento del centroide
                    if self.centroide_anterior is not None:
                        cv2.line(frame, self.centroide_anterior, centroide_actual, (0, 0, 255), 2)
                    self.centroide_anterior = centroide_actual

        return frame


# Ejemplo de uso
detector = DetectorDeObstaculos()
video_capture = cv2.VideoCapture('/Users/lichi/Desktop/Tesis/detector-obstaculos-logica-difusa/img/LANCHA_RC.mp4')  # Capturamos video desde la cámara

while True:
    ret, frame = video_capture.read()
    frame = detector.detectar_obstaculo(frame)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
