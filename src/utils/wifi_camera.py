import cv2

# Abre la cámara
rstp_url = "rtsp://192.168.1.1:554/live"
cap = cv2.VideoCapture(rstp_url)  # 0 generalmente es el índice de la cámara USB

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir el frame (final de video). Saliendo...")
        break

    # Muestra el frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara
cap.release()
cv2.destroyAllWindows()
