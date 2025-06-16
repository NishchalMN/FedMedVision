import threading, time, uuid
from datetime import datetime
from aggregator_service.db.database import SessionLocal
from aggregator_service.db import crud

REQUIRED_CLIENTS = 3

def selector():
    while True:
        time.sleep(60)
        db = SessionLocal()
        ready = crud.get_ready_clients(db)
        if len(ready) >= REQUIRED_CLIENTS:
            selected = crud.select_clients(db, REQUIRED_CLIENTS)
            round_id = str(uuid.uuid4())[:8]
            crud.create_round(db, round_id, selected)
            print(f"🚀 Started new round {round_id} with {[c.client_id for c in selected]}")
        db.close()

def start_scheduler():
    threading.Thread(target=selector, daemon=True).start()
    print("📡 Round selector thread started")