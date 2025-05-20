import face_recognition
import numpy as np
from PIL import Image
import io

def verificar_rostros(ine_image_file, camera_image_file, threshold=0.6):
    if hasattr(ine_image_file, 'temporary_file_path'):
        image_ine = face_recognition.load_image_file(ine_image_file.temporary_file_path())
    else:
        image_ine = face_recognition.load_image_file(ine_image_file)

    face_locations_ine = face_recognition.face_locations(image_ine)
    if len(face_locations_ine) == 0:
        return {"error": "No se detectó rostro en la INE."}, 400
    face_encoding_ine = face_recognition.face_encodings(image_ine, known_face_locations=face_locations_ine)[0]

    image_bytes = camera_image_file.read()
    image_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    face_locations_cam = face_recognition.face_locations(image_np)
    if len(face_locations_cam) == 0:
        return {"error": "No se detectó rostro en la imagen de la cámara."}, 400
    face_encoding_cam = face_recognition.face_encodings(image_np, known_face_locations=face_locations_cam)[0]

    top, right, bottom, left = face_locations_cam[0]
    face_width = right - left
    frame_width = image_np.shape[1]
    face_ratio = face_width / frame_width

    if face_ratio > 0.6:
        distancia_msg = "Rostro demasiado cerca. Aléjate un poco."
    elif face_ratio < 0.15:
        distancia_msg = "Rostro muy lejos. Acércate un poco."
    else:
        distancia_msg = "Distancia adecuada."

    dist = np.linalg.norm(face_encoding_ine - face_encoding_cam)
    is_match = dist < threshold

    return {
        "match": is_match,
        "distancia": round(dist, 4),
        "sugerencia": distancia_msg
    }, 200
