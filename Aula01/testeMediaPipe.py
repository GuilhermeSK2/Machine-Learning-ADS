#https://github.com/google-ai-edge/mediapipe


import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Inicializa a captura da webcam
cap = cv2.VideoCapture(0)

# Inicializa a detecção facial com MediaPipe
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando frames em branco.")
            continue

        # Conversão da imagem para RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Aplicando a imagem na rede para detectar faces
        results = face_detection.process(image_rgb)

        # Se detectar rostos, desenha na imagem original
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Apresentando o resultado na tela
        cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))

        # Se a tecla "ESC" for pressionada, sai do loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
