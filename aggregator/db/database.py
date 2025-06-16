from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./fedmedvision.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    from . import models
    Base.metadata.create_all(bind=engine)

    # Seed clients if not present
    from .database import SessionLocal
    db = SessionLocal()
    for client_id, token in {
        "client_1": "token_abc123",
        "client_2": "token_def456",
        "client_3": "token_xyz789"
    }.items():
        if not db.query(models.Client).filter_by(client_id=client_id).first():
            db.add(models.Client(client_id=client_id, token=token))
    db.commit()