from sqlalchemy import Column, String, DateTime, Boolean, JSON, Enum
from sqlalchemy.ext.mutable import MutableList
from .database import Base
import enum
from datetime import datetime, timezone

class ClientStatus(str, enum.Enum):
    idle = "idle"
    ready = "ready"
    selected = "selected"
    submitted = "submitted"

class RoundStatus(str, enum.Enum):
    started = "started"
    aggregated = "aggregated"

class Client(Base):
    __tablename__ = "clients"

    client_id = Column(String, primary_key=True, index=True)
    token = Column(String)
    last_ready = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(ClientStatus), default=ClientStatus.idle)
    last_round_id = Column(String, nullable=True)

class Round(Base):
    __tablename__ = "rounds"

    round_id = Column(String, primary_key=True, index=True)
    start_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_time = Column(DateTime, nullable=True)
    # selected_clients = Column(JSON, default=[])
    # submitted_clients = Column(JSON, default=[])
    
    selected_clients = Column(MutableList.as_mutable(JSON), default=list)
    submitted_clients = Column(MutableList.as_mutable(JSON), default=list)
    status = Column(Enum(RoundStatus), default=RoundStatus.started)
    metrics = Column(JSON, nullable=True)

