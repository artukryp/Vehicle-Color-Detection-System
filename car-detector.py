import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('audio-car.MP4')

# Obtener propiedades del video original
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear VideoWriter para guardar el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
out = cv2.VideoWriter('salida_car_det.mp4', fourcc, fps, (width, height))

VEHICULOS = {'car', 'truck', 'bus', 'motorcycle'}
CONF_THRESHOLD = 0.5
COLOR_MIN_PORC = 0.3  # 30% píxeles dentro del rango para considerar válido

azul_oscuro_bajo = np.array([90, 70, 30])
azul_oscuro_alto = np.array([130, 255, 80])

negro_bajo = np.array([0, 0, 0])
negro_alto = np.array([180, 255, 60])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for box, score, cls in zip(boxes, scores, classes):
        label = model.names[cls]
        if label not in VEHICULOS or score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_azul = cv2.inRange(hsv, azul_oscuro_bajo, azul_oscuro_alto)
        mask_negro = cv2.inRange(hsv, negro_bajo, negro_alto)

        porc_azul = (mask_azul > 0).sum() / mask_azul.size
        porc_negro = (mask_negro > 0).sum() / mask_negro.size

        if porc_azul > COLOR_MIN_PORC or porc_negro > COLOR_MIN_PORC:
            color = (0, 0, 255)
            texto = "Black Car"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, texto, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Escribir frame procesado en el archivo
    out.write(frame)

    cv2.imshow('Autos Azul Oscuro o Negro', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
