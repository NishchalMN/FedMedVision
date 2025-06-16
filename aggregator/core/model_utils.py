from torchvision.models import resnet18
from torch import nn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import Set, Dict
import os, shutil, time, threading, uuid
import torch
import uvicorn
from datetime import datetime
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18
import torch.nn as nn
from fastapi import Header, HTTPException
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import mlflow
import mlflow.pytorch

VAL_CSV = "/home/nishchal/Final_set/validation/global_validation.csv"

# Dataset for validation
class ValidationDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {"NORMAL": 0, "PNEUMONIA": 1, "COVID": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = self.label_map[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model

def evaluate_model(model_path, csv_path=VAL_CSV, round_id=None, save_dir="metrics"):
    os.makedirs(save_dir, exist_ok=True)

    model = get_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = ValidationDataset(csv_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            out = model(x)
            preds = out.argmax(dim=1).tolist()
            y_true += y.tolist()
            y_pred += preds

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "round_id": round_id or "unknown",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),  # to make it JSON serializable
    }

    # Save confusion matrix plot
    cm_plot_path = os.path.join(save_dir, f"confusion_matrix_{round_id or 'latest'}.png")
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Pneumonia", "COVID"],
                    yticklabels=["Normal", "Pneumonia", "COVID"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (Round {round_id})" if round_id else "Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"✅ Saved {cm_plot_path}")
        metrics["confusion_matrix_path"] = cm_plot_path
    except Exception as e:
        print("⚠️ Could not save confusion matrix:", e)

    # Optional: log to JSON
    log_path = os.path.join(save_dir, f"metrics_round_{round_id or 'latest'}.json")
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📝 Logged metrics to {log_path}")

    return metrics
