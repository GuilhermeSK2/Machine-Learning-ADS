import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Configuração do desenho
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Captura da webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando frames em branco.")
            continue

        # Conversão para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Aplicando a imagem à rede
        results = face_mesh.process(image)

        # Retorna para BGR para exibição
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Desenho dos pontos da face
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Exibir resultado
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        # Se pressionar ESC, sai do loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Liberar a câmera e fechar janelas
cap.release()
cv2.destroyAllWindows()
