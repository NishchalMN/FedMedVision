import requests
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from PIL import Image
from io import BytesIO
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CLIENT_ID = os.environ.get("CLIENT_ID", "client_1")
CLIENT_TOKEN = os.environ.get("CLIENT_TOKEN", "token_abc123")
AGGREGATOR_URL = os.environ.get("AGGREGATOR_URL", "http://40.76.121.167:8008")
CSV_PATH = f"client_data.csv"
DATA_PATH = Path("/app/data")

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
TMP_MODEL_PATH = "./tmp_model.pt"

# Model (ResNet18 fine-tuned for 3-class classification)
from torchvision.models import resnet18


def get_model():
    model = resnet18(weights=True)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model


# Custom Dataset
class XrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {"NORMAL": 0, "PNEUMONIA": 1, "COVID": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(DATA_PATH / row["image_path"]).convert("RGB")
        label = self.label_map[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def mark_ready():
    headers = {"Authorization": f"Bearer {CLIENT_TOKEN}"}
    data = {"client_id": CLIENT_ID}
    res = requests.post(f"{AGGREGATOR_URL}/ready-to-train", data=data, headers=headers)
    logger.info("Marked ready" if res.ok else res.text)


def wait_until_selected():
    while True:
        headers = {"Authorization": f"Bearer {CLIENT_TOKEN}"}
        res = requests.get(
            f"{AGGREGATOR_URL}/can-start-round",
            params={"client_id": CLIENT_ID},
            headers=headers,
        )
        logger.info("Checking if selected for training...")
        if res.ok and res.json().get("start"):
            round_id = res.json()["round_id"]
            logger.info(f"Selected for round {round_id}")
            return round_id
        time.sleep(60)


def download_model():
    res = requests.get(f"{AGGREGATOR_URL}/global-model")
    if res.ok:
        with open(TMP_MODEL_PATH, "wb") as f:
            f.write(res.content)
    else:
        raise Exception("Failed to download model")


def train_model():
    model = get_model()
    model.load_state_dict(torch.load(TMP_MODEL_PATH))
    logger.info("Loaded model from aggregator and starting training")
    model.train()

    dataset = XrayDataset(CSV_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(LOCAL_EPOCHS):
        for images, labels in tqdm(loader):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def submit_update(state_dict):
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    files = {"file": ("update.pt", buffer, "application/octet-stream")}
    data = {"client_id": CLIENT_ID}
    headers = {"Authorization": f"Bearer {CLIENT_TOKEN}"}
    res = requests.post(
        f"{AGGREGATOR_URL}/submit-update", data=data, headers=headers, files=files
    )
    logger.info("Submitted update" if res.ok else res.text)


if __name__ == "__main__":
    for epoch in range(5):
        logger.info(f"Starting epoch {epoch + 1}")
        mark_ready()
        wait_until_selected()
        download_model()
        updated_state = train_model()
        submit_update(updated_state)
