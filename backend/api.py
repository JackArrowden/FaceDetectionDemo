import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import sys
import os

# Add YOLOFaceV2 to sys.path
sys.path.append(os.path.abspath("YOLOFaceV2"))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox
import io

import tempfile
import base64

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Parameter configuration
WEIGHTS = "best.pt"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = ""

# Model initialization
device = select_device(DEVICE)
model = attempt_load(WEIGHTS, map_location=device)
stride = int(model.stride.max()) * 2
imgsz = check_img_size(IMG_SIZE, s=stride)
half = device.type != "cpu"
if half:
    model.half()
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

# Start model
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

# Image processing function
def process_image(img0):
    # Image preprocessing
    img = letterbox(img0, new_shape=imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)

    # Post-processing
    num_faces = 0
    if pred[0] is not None and len(pred[0]):
        num_faces = len(pred[0])  # Số lượng khuôn mặt
        pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], img0.shape).round()
        for *xyxy, conf, cls in pred[0]:
            label = f"{names[int(cls)]} {conf:.2f}"
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)

    return img0, num_faces

# File type checking function
def is_image_file(filename: str) -> bool:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return os.path.splitext(filename.lower())[1] in image_extensions

# API endpoint for file detection
@app.post("/detect")
async def detect_file(file: UploadFile = File(...)):
    # File reading
    file_bytes = await file.read()
    filename = file.filename

    # File type checking based on extension
    if is_image_file(filename):
        # Image processing
        nparr = np.frombuffer(file_bytes, np.uint8)
        img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img0 is None:
            return JSONResponse(content={"error": "Không thể đọc file ảnh"}, status_code=400)

        # Image processing
        img_result, num_faces = process_image(img0)

        # Convert image to base64
        _, img_encoded = cv2.imencode(".jpg", img_result)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        # Return JSON response with image and number of faces
        return JSONResponse(content={
            "type": "image",
            "data": img_base64,
            "num_faces": num_faces
        })

    else:
        # Process video file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(file_bytes)
            temp_input_path = temp_input.name

        # Open video
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            os.remove(temp_input_path)
            return JSONResponse(content={"error": "Không thể mở video"}, status_code=400)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            temp_output_path = temp_output.name

        # Intialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # Process video frames
        total_num_faces = 0
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame
            processed_frame, num_faces = process_image(frame)
            out.write(processed_frame)
            total_num_faces += num_faces
            frame_count += 1

        # Resource cleanup
        cap.release()
        out.release()

        # Read video file and convert to base64
        with open(temp_output_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

        # Clear temporary files
        os.remove(temp_input_path)
        os.remove(temp_output_path)

        # Return JSON response with video and average number of faces
        avg_num_faces = total_num_faces / frame_count if frame_count > 0 else 0
        return JSONResponse(content={
            "type": "video",
            "data": video_base64,
            "num_faces": avg_num_faces
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)