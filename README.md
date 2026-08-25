# Crowd Guardian!  - Deployment and Setup Guide

## Requirements
- Python 3.9+
- Node.js 18+
- Ollama (running locally with `qwen3:8b` pulled)

## 1. Start the Backend

1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   *Note: On first run, YOLOv8 weights (`yolov8n.pt`) will automatically download.*

## 2. Start the Frontend

1. Open a new terminal and navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies (if not already done):
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Access the dashboard at the URL provided by Vite (usually `http://localhost:5173`).

## 3. Verify Ollama (Agent 6)

Ensure you have Ollama installed and the model is pulled:
```bash
ollama run qwen3:8b
```
Keep the Ollama service running in the background. If it is offline, the Advisor Agent will gracefully fallback to a default error message, but the rest of the pipeline will still function.
