# FedMedVision
 
A federated learning platform for medical image classification, built with PyTorch and FastAPI. FedMedVision enables distributed training of chest X-ray classifiers across multiple client nodes without sharing raw patient data - preserving privacy while building accurate diagnostic models.
 
## Overview
 
FedMedVision implements a centralized **Federated Averaging (FedAvg)** protocol where:
 
- A central **aggregator** server orchestrates training rounds and maintains the global model
- Multiple **client trainers** perform local training on their private datasets
- Only model weights are exchanged — raw data never leaves the client
 
The system classifies chest X-rays into three categories: **Normal**, **Pneumonia**, and **COVID**.
 
## Architecture
 
```
┌─────────────────────────────────────────────────┐
│                  Aggregator (FastAPI)            │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐  │
│  │ Scheduler  │  │ FedAvg     │  │ MLflow    │  │
│  │ (rounds)   │  │ Aggregation│  │ Tracking  │  │
│  └───────────┘  └────────────┘  └───────────┘  │
│  ┌───────────┐  ┌────────────┐                  │
│  │ REST API   │  │ SQLite DB  │                  │
│  └───────────┘  └────────────┘                  │
└──────────┬──────────┬──────────┬────────────────┘
           │          │          │
      ┌────┘     ┌────┘     ┌───┘
      ▼          ▼          ▼
 ┌─────────┐ ┌─────────┐ ┌─────────┐
 │Client 1  │ │Client 2  │ │Client 3  │
 │(Trainer) │ │(Trainer) │ │(Trainer) │
 │Local Data│ │Local Data│ │Local Data│
 └─────────┘ └─────────┘ └─────────┘
```
 
## Project Structure
 
```
FedMedVision/
├── aggregator/                 # Central server
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── routes_clients.py   # Client lifecycle endpoints
│   │   └── routes_model.py     # Global model serving endpoints
│   ├── core/
│   │   ├── aggregator.py       # FedAvg weight aggregation + MLflow logging
│   │   ├── fl_scheduler.py     # Round scheduling (selects clients every 60s)
│   │   └── model_utils.py      # ResNet-18 model creation & evaluation
│   ├── db/
│   │   ├── database.py         # SQLAlchemy setup (SQLite)
│   │   ├── models.py           # Client & Round database schemas
│   │   └── crud.py             # Database operations
│   ├── requirements.txt
│   └── Dockerfile
├── trainer/                    # Client training node
│   ├── trainer.py              # Main client training loop
│   ├── data.py                 # Dataset loading (local, URL, S3)
│   ├── docker-compose.yml
│   └── Dockerfile
└── LICENSE                     # Apache 2.0
```
 
## How It Works
 
### Training Round Lifecycle
 
1. **Clients signal readiness** — each client calls `POST /ready-to-train`
2. **Scheduler creates a round** — when 3+ clients are ready, the scheduler selects them
3. **Clients download the global model** — via `GET /global-model`
4. **Local training** — each client trains ResNet-18 on its local chest X-ray data (1 epoch, Adam optimizer)
5. **Clients submit updated weights** — via `POST /submit-update`
6. **Aggregation** — the server averages all client weights (FedAvg) and updates the global model
7. **Evaluation** — the aggregated model is evaluated on a held-out validation set
8. **Logging** — metrics (accuracy, F1, precision, recall, confusion matrix) are logged to MLflow
 
### API Endpoints
 
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/ready-to-train` | Client marks itself as ready |
| `GET` | `/can-start-round` | Client checks if selected for a round |
| `GET` | `/global-model` | Download the current global model |
| `POST` | `/submit-update` | Submit locally trained weights |
 
All client endpoints require a `Bearer` token for authentication.
 
## Getting Started
 
### Prerequisites
 
- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- At least 3 client nodes with local chest X-ray datasets
 
### 1. Start the Aggregator
 
```bash
cd aggregator
docker build -t fedmedvision-aggregator .
docker run -d -p 8000:8000 -v /mnt/shared:/mnt/shared fedmedvision-aggregator
```
 
Or run directly:
 
```bash
cd aggregator
pip install -r requirements.txt
uvicorn aggregator_service.main:app --host 0.0.0.0 --port 8000
```
 
### 2. Start Client Trainers
 
Each client needs a CSV file (`client_data.csv`) with columns `image_path` and `label`, where labels are one of: `NORMAL`, `PNEUMONIA`, `COVID`.
 
```bash
cd trainer
docker-compose build
docker-compose up
```
 
Or run directly with environment variables:
 
```bash
export CLIENT_ID=client_1
export CLIENT_TOKEN=token_abc123
export AGGREGATOR_URL=http://<aggregator-host>:8000
python trainer.py
```
 
### 3. Monitor Training
 
Metrics are tracked via MLflow. Access the MLflow UI at `http://localhost:8080` to view:
 
- Per-round accuracy, precision, recall, and F1 scores
- Confusion matrices
- Registered global models
 
## Data Format
 
Client datasets must be provided as a CSV with the following structure:
 
| image_path | label |
|---|---|
| `/data/images/patient_001.png` | NORMAL |
| `/data/images/patient_002.png` | PNEUMONIA |
| `/data/images/patient_003.png` | COVID |
 
Images are resized to 224x224 and normalized during training.
 
## Model
 
- **Architecture**: ResNet-18 (torchvision) with a 3-class output layer
- **Input**: 224x224 RGB images
- **Classes**: Normal (0), Pneumonia (1), COVID (2)
- **Local training**: 1 epoch per round, batch size 32, Adam optimizer (lr=1e-5)
- **Aggregation**: Federated Averaging (FedAvg)
 
## Tech Stack
 
- **Deep Learning**: PyTorch, torchvision
- **Server**: FastAPI, Uvicorn
- **Database**: SQLite via SQLAlchemy
- **Experiment Tracking**: MLflow
- **Metrics**: scikit-learn
- **Deployment**: Docker, Docker Compose
 
## License
 
This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.