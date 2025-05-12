# Face Detection Demo Setup Guide

## Prerequisites
- Node.js and npm installed
- Python 3.x installed
- Git installed

## Setup Instructions

1. **Frontend Setup**
   - Navigate to the frontend directory:
     ```bash
     cd frontend
     ```
   - Install dependencies:
     ```bash
     npm install
     ```
   - Fix any vulnerabilities if needed:
     ```bash
     npm audit fix
     ```

2. **Backend Setup**
   - Navigate to the backend directory:
     ```bash
     cd backend
     ```
   - Initialize git submodule:
     ```bash
     git submodule init
     ```
   - Modify the YOLOFaceV2 model file:
     - Open `FaceDetectionDemo/backend/YOLOFaceV2/models/experimental.py`
     - At line 118, update `torch.load` to include `weights_only=False`:
       ```python
       ckpt = torch.load(w, map_location=map_location, weights_only=False)  # load
       ```
   - Download and extract OpenH264 library:
     - Visit [OpenH264 v1.8.0 release](https://github.com/cisco/openh264/releases/tag/v1.8.0)
     - Download and extract the compatible library to the backend directory.

3. **Run Frontend**
   - Navigate to the frontend directory:
     ```bash
     cd frontend
     ```
   - Start the development server:
     ```bash
     npm run dev
     ```

4. **Run Backend**
   - Navigate to the backend directory:
     ```bash
     cd backend
     ```
   - Run the API:
     ```bash
     python api.py
     ```

## Notes
- Ensure all dependencies are installed correctly before running.
- The frontend and backend must both be running to use the application.