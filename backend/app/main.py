import os
import cv2
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.database import database, models
from app.agents.graph import app_graph
from app.agents.state import CrowdState

# Create database tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Crowd Guardian API")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images statically
os.makedirs("data/images", exist_ok=True)
app.mount("/images", StaticFiles(directory="data/images"), name="images")

@app.get("/")
def root():
    return {"message": "Welcome to Crowd Guardian API"}

@app.post("/api/analyze")
async def analyze_crowd_image(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only images are allowed")
        
    # Generate unique filename to avoid overwrites
    ext = os.path.splitext(file.filename)[1]
    safe_filename = f"{uuid.uuid4()}{ext}"
    local_path = os.path.join("data", "images", safe_filename)
    
    # Save the file locally
    with open(local_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Read image into OpenCV format
    image_array = cv2.imread(local_path)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
        
    # Initialize LangGraph State
    initial_state = CrowdState(
        image_path=local_path,
        image_array=image_array,
        people_count=0, density_score=0.0, bounding_boxes=[],
        zone_counts={}, hotspot_zone="", max_zone_count=0,
        heatmap_array=None, heatmap_image_path="",
        risk_level="", congestion_score=0, hotspot_concentration=0.0,
        future_people_count=0, stampede_risk=0,
        alert_level="", warning_message="", requires_intervention=False,
        llm_risk_summary="", llm_safety_recommendations="", llm_incident_report="",
        errors=[]
    )
    
    # Run the graph workflow
    final_state = app_graph.invoke(initial_state)
    
    if final_state.get("errors"):
        # Log errors but still try to return what we have
        print("Graph execution errors:", final_state["errors"])
        
    # Save results to database
    db_record = models.AnalysisRecord(
        original_image_path=local_path,
        heatmap_image_path=final_state.get("heatmap_image_path", ""),
        people_count=final_state.get("people_count", 0),
        density_score=final_state.get("density_score", 0.0),
        hotspot_zone=final_state.get("hotspot_zone"),
        max_zone_count=final_state.get("max_zone_count", 0),
        risk_level=final_state.get("risk_level"),
        congestion_score=final_state.get("congestion_score", 0),
        future_people_count=final_state.get("future_people_count", 0),
        stampede_risk=final_state.get("stampede_risk", 0),
        alert_level=final_state.get("alert_level"),
        warning_message=final_state.get("warning_message"),
        requires_intervention=final_state.get("requires_intervention", False),
        llm_risk_summary=final_state.get("llm_risk_summary"),
        llm_safety_recommendations=final_state.get("llm_safety_recommendations"),
        llm_incident_report=final_state.get("llm_incident_report")
    )
    
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    # Build clean JSON response
    response_data = {
        "id": db_record.id,
        "telemetry": {
            "people_count": db_record.people_count,
            "density_score": db_record.density_score,
            "hotspot_zone": db_record.hotspot_zone,
            "max_zone_count": db_record.max_zone_count,
        },
        "risk": {
            "risk_level": db_record.risk_level,
            "congestion_score": db_record.congestion_score,
            "future_people_count": db_record.future_people_count,
            "stampede_risk": db_record.stampede_risk,
        },
        "alert": {
            "alert_level": db_record.alert_level,
            "warning_message": db_record.warning_message,
            "requires_intervention": db_record.requires_intervention,
        },
        "advisor": {
            "summary": db_record.llm_risk_summary,
            "recommendations": db_record.llm_safety_recommendations,
            "incident_report": db_record.llm_incident_report,
        },
        "images": {
            "original": f"/{local_path.replace(os.sep, '/')}",
            "heatmap": f"/{db_record.heatmap_image_path.replace(os.sep, '/')}" if db_record.heatmap_image_path else None
        },
        "errors": final_state.get("errors")
    }
    
    return response_data

@app.get("/api/history")
def get_history(limit: int = 10, db: Session = Depends(database.get_db)):
    records = db.query(models.AnalysisRecord).order_by(models.AnalysisRecord.created_at.desc()).limit(limit).all()
    return records
