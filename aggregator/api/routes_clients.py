from fastapi import APIRouter, Form, Header, UploadFile, File, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from aggregator_service.db import database, crud
from aggregator_service.db.models import ClientStatus, Client
import shutil, os
import torch
from aggregator_service.core.aggregator import aggregate_weights

router = APIRouter()

MODEL_DIR = "/mnt/shared"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/ready-to-train")
def client_ready(client_id: str = Form(...), authorization: str = Header(..., alias="Authorization"), db: Session = Depends(get_db)):
    token = authorization.replace("Bearer ", "")
    client = db.query(Client).filter_by(client_id=client_id, token=token).first()
    if not client:
        raise HTTPException(status_code=401, detail="Unauthorized")

    crud.set_client_ready(db, client_id)
    print(f"✅ {client_id} marked ready in DB")
    return {"status": f"{client_id} marked as ready"}

@router.get("/can-start-round")
def can_start(client_id: str, authorization: str = Header(..., alias="Authorization"), db: Session = Depends(get_db)):
    token = authorization.replace("Bearer ", "")
    client = db.query(Client).filter_by(client_id=client_id, token=token).first()
    if not client:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if client.status == ClientStatus.selected:
        return {"start": True, "round_id": client.last_round_id}
    return {"start": False}

@router.post("/submit-update")
async def submit_update(client_id: str = Form(...), authorization: str = Header(..., alias="Authorization"), file: UploadFile = File(...), background_tasks: BackgroundTasks = None, db: Session = Depends(get_db)):
    token = authorization.replace("Bearer ", "")
    client = db.query(Client).filter_by(client_id=client_id, token=token).first()
    if not client:
        raise HTTPException(status_code=401, detail="Unauthorized")

    save_path = os.path.join(MODEL_DIR, f"update_{client_id}.pt")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    crud.mark_client_submitted(db, client_id)
    background_tasks.add_task(aggregate_weights)

    print(f"📥 Local weights received from {client_id}")
    return {"status": f"Update received from {client_id}"}


# from fastapi import APIRouter, Form, Header, UploadFile, File, BackgroundTasks, Depends, HTTPException
# from sqlalchemy.orm import Session
# from aggregator_service.db import database, crud
# from aggregator_service.db.models import ClientStatus
# import shutil, os, uuid
# import torch
# from aggregator_service.core.aggregator import aggregate_weights

# router = APIRouter()

# MODEL_DIR = "/mnt/shared"
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Dummy client tokens (replace with DB validation later)
# VALID_CLIENTS = {
#     "client_1": "token_abc123",
#     "client_2": "token_def456",
#     "client_3": "token_xyz789"
# }

# def get_db():
#     db = database.SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @router.post("/ready-to-train")
# def client_ready(client_id: str = Form(...), authorization: str = Header(..., alias="Authorization"), db: Session = Depends(get_db)):
#     token = authorization.replace("Bearer ", "")
#     if VALID_CLIENTS.get(client_id) != token:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     crud.set_client_ready(db, client_id)
#     return {"status": f"{client_id} marked as ready"}

# @router.get("/can-start-round")
# def can_start(client_id: str, authorization: str = Header(..., alias="Authorization"), db: Session = Depends(get_db)):
#     token = authorization.replace("Bearer ", "")
#     if VALID_CLIENTS.get(client_id) != token:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     client = db.query(crud.models.Client).filter_by(client_id=client_id).first()
#     if client and client.status == ClientStatus.selected:
#         return {"start": True, "round_id": client.last_round_id}
#     return {"start": False}

# @router.post("/submit-update")
# async def submit_update(client_id: str = Form(...), authorization: str = Header(..., alias="Authorization"), file: UploadFile = File(...), background_tasks: BackgroundTasks = None, db: Session = Depends(get_db)):
#     token = authorization.replace("Bearer ", "")
#     if VALID_CLIENTS.get(client_id) != token:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     save_path = os.path.join(MODEL_DIR, f"update_{client_id}.pt")
#     with open(save_path, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     crud.mark_client_submitted(db, client_id)

#     # Trigger aggregation check
#     background_tasks.add_task(aggregate_weights)

#     return {"status": f"Update received from {client_id}"}

