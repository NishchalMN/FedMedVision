from sqlalchemy.orm import Session
from . import models
from aggregator_service.db.models import Client, Round
from datetime import datetime

def get_ready_clients(db: Session):
    return db.query(models.Client).filter(models.Client.status == models.ClientStatus.ready).all()

def select_clients(db: Session, num_required: int):
    ready = get_ready_clients(db)
    return ready[:num_required]

def create_round(db: Session, round_id: str, selected_clients: list):
    db_round = models.Round(
        round_id=round_id,
        selected_clients=[c.client_id for c in selected_clients]
    )
    db.add(db_round)
    for client in selected_clients:
        client.status = models.ClientStatus.selected
        client.last_round_id = round_id
    db.commit()
    return db_round

def update_round_metrics(db: Session, round_id: str, metrics: dict):
    db_round = db.query(models.Round).filter(models.Round.round_id == round_id).first()
    if db_round:
        db_round.metrics = metrics
        db_round.status = models.RoundStatus.aggregated
        db_round.end_time = datetime.utcnow()
        db.commit()


def set_client_ready(db: Session, client_id: str):
    client = db.query(models.Client).filter(models.Client.client_id == client_id).first()
    # print(client.last_round_id)
    if client:
        client.status = models.ClientStatus.ready
        client.last_ready = datetime.utcnow()
        db.commit()


def mark_client_submitted(db: Session, client_id: str):
    client = db.query(Client).filter_by(client_id=client_id).first()
    # print(client.last_round_id)
    if not client or not client.last_round_id:
        return

    round_obj = db.query(models.Round).filter_by(round_id=client.last_round_id).first()
    if round_obj and client_id not in round_obj.submitted_clients:
        round_obj.submitted_clients.append(client_id)
        db.commit()
