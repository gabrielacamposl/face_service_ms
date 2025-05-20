from fastapi import FastAPI, UploadFile, File
from app.face_utils import verificar_rostros

app = FastAPI()

@app.post("/verificar")
async def verificar(ine_image: UploadFile = File(...), camera_image: UploadFile = File(...)):
    result, status_code = verificar_rostros(ine_image.file, camera_image.file)
    return result
