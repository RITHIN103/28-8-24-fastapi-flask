from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json

from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
from predictFromModel import prediction

# Set environment variables
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define models
class FilePath(BaseModel):
    filepath: str

class FolderPath(BaseModel):
    folderPath: str

@app.get("/")
async def home():
    # Return an HTML page; for FastAPI, this requires a separate static file server setup.
    return JSONResponse(content={"message": "Home page. Static file handling is required."})

@app.post("/predict")
async def predict_route_client(data: FilePath):
    try:
        path = data.filepath
        
        pred_val = pred_validation(path)
        pred_val.prediction_validation()
        
        pred = prediction(path)
        path, json_predictions = pred.predictionFromModel()
        
        return JSONResponse(
            content={
                "message": f"Prediction File created at {path}",
                "predictions": json.loads(json_predictions)
            }
        )
    except (ValueError, KeyError) as e:
        return JSONResponse(status_code=400, content={"error": f"Error Occurred: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error Occurred: {str(e)}"})

@app.post("/train")
async def train_route_client(data: FolderPath):
    try:
        path = data.folderPath
        
        train_val_obj = train_validation(path)
        train_val_obj.train_validation()
        
        train_model_obj = trainModel()
        train_model_obj.trainingModel()
        
        return JSONResponse(content={"message": "Training successful!"})
    except (ValueError, KeyError) as e:
        return JSONResponse(status_code=400, content={"error": f"Error Occurred: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error Occurred: {str(e)}"})

# Run the server
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
