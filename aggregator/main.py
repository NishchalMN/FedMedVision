from fastapi import FastAPI
from aggregator_service.api import routes_clients, routes_model
from aggregator_service.core.fl_scheduler import start_scheduler
from aggregator_service.db.database import init_db
from aggregator_service.core.model_utils import get_model
import torch, os

MODEL_PATH = "/mnt/shared/global_model.pt"
os.makedirs("/mnt/shared", exist_ok=True)

# Save a fresh global model if missing
if not os.path.exists(MODEL_PATH):
    print("=> No global model found. Initializing fresh one...")
    torch.save(get_model().state_dict(), MODEL_PATH)

app = FastAPI()

# Include routers
app.include_router(routes_clients.router)
app.include_router(routes_model.router)

# On startup, initialize DB and scheduler
@app.on_event("startup")
def startup_event():
    init_db()
    start_scheduler()

# Optional root route
@app.get("/")
def root():
    return {"message": "FedMedVision Aggregator API"}
