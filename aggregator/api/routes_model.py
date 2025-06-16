from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
import mlflow.pytorch
import torch
import os

router = APIRouter()

TMP_MODEL_PATH = "/mnt/shared/global_model.pt"
MLFLOW_TRACKING_URI = "http://fedadmin:fedmed@localhost:8080"

@router.get("/global-model")
def get_model():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Try to fetch the latest registered model from MLflow
        model = mlflow.pytorch.load_model("models:/FedMedGlobalModel/None")
        torch.save(model.state_dict(), TMP_MODEL_PATH)

        return FileResponse(TMP_MODEL_PATH, media_type="application/octet-stream", filename="global_model.pt")

    except Exception as e:
        # print(f"⚠️ Failed to fetch from MLflow: {e}")
        
        # Try local fallback
        if os.path.exists(TMP_MODEL_PATH):
            # print("📁 Falling back to local model in /mnt/shared")
            return FileResponse(TMP_MODEL_PATH, media_type="application/octet-stream", filename="global_model.pt")
        else:
            return JSONResponse({"error": "No global model found in MLflow or local disk."}, status_code=404)