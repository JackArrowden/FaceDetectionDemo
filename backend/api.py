import cv2
import torch
import numpy as np
import base64
import tempfile
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add YOLOFaceV2 to sys.path
import sys
sys.path.append(os.path.abspath("YOLOFaceV2"))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox

def create_app():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    face_detector = FaceDetector()
    file_processor = FileProcessor(face_detector)
    app.state.file_processor = file_processor

    @app.post("/detect")
    async def detect_file(file: UploadFile = File(...)):
        file_bytes = await file.read()
        filename = file.filename

        if app.state.file_processor.is_image_file(filename):
            result, error = app.state.file_processor.process_image_file(file_bytes)
        elif app.state.file_processor.is_video_file(filename):
            result, error = app.state.file_processor.process_video_file(file_bytes)
        else:
            return JSONResponse(content={"error": "Unsupported file type"}, status_code=403)

        if error:
            return JSONResponse(content={"error": error}, status_code=400)
        return JSONResponse(content=result)

    return app

class FaceDetector:
    def __init__(self, weights="best.pt", img_size=640, conf_thres=0.5, iou_thres=0.45, device=""):
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max()) * 2
        self.imgsz = check_img_size(img_size, s=self.stride)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.half = self.device.type != "cpu"
        if self.half:
            self.model.half()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

    def process_image(self, img0):
        img = letterbox(img0, new_shape=self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        num_faces = 0
        if pred[0] is not None and len(pred[0]):
            num_faces = len(pred[0])
            pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred[0]:
                label = f"{self.names[int(cls)]} {conf:.2f}"
                plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)

        return img0, num_faces

class FileProcessor:
    def __init__(self, detector: FaceDetector):
        self.detector = detector

    def is_image_file(self, filename: str) -> bool:
        return os.path.splitext(filename.lower())[-1] in {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo'}
    
    def is_video_file(self, filename: str) -> bool:
        return os.path.splitext(filename.lower())[-1] in {'.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv'}

    def process_image_file(self, file_bytes: bytes):
        nparr = np.frombuffer(file_bytes, np.uint8)
        img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img0 is None:
            return None, "The image file can not be read"
        result_img, num_faces = self.detector.process_image(img0)
        _, img_encoded = cv2.imencode(".jpg", result_img)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")
        return {"type": "image", "data": img_base64, "num_faces": num_faces}, None

    def process_video_file(self, file_bytes: bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(file_bytes)
            input_path = temp_input.name

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            os.remove(input_path)
            return None, "The video can not be openned"

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            output_path = temp_output.name

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

        total_faces = 0
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, num_faces = self.detector.process_image(frame)
            out.write(processed_frame)
            total_faces += num_faces
            frame_count += 1

        cap.release()
        out.release()
        os.remove(input_path)

        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(output_path)

        return {
            "type": "video",
            "data": video_base64,
            "num_faces": total_faces / frame_count if frame_count > 0 else 0
        }, None

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)