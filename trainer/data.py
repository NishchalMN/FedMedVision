# dataset.py

import json
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

import boto3
from fastai.vision.all import *
from fastai.data.external import untar_data

DEFAULT_DATA_PATH = Path("/data")


class Dataset:
    def __init__(self, config: dict, logger=None):
        """
        config should have at top level:
          - 'data' → dict with 'data_source', 'classes', 'transforms'
          - 'train' → dict with training params
        """
        self.logger = logger
        self.config = config

        # Pull data_source config out of the 'data' block
        ds_cfg = config["data"]["data_source"]
        ds_type = ds_cfg["type"].lower()

        if ds_type == "local":
            # Optionally allow custom local path
            self.data_path = Path(ds_cfg.get("path", DEFAULT_DATA_PATH))
        elif ds_type == "url":
            # untar_data will download & extract
            self.data_path = untar_data(ds_cfg["url"])
        elif ds_type == "s3":
            # download then extract under DEFAULT_DATA_PATH
            self.data_path = DEFAULT_DATA_PATH
            creds = ds_cfg.get("credentials", {})
            self._download_s3_data(
                ds_cfg["path"],
                creds.get("access_key"),
                creds.get("secret_key"),
                self.data_path,
            )
        else:
            raise ValueError(f"Unsupported data source type: {ds_type}")

        # Build the dataloaders immediately
        self.dls = self.create_dataloaders()

    def _download_s3_data(
        self,
        s3_uri: str,
        access_key: Optional[str],
        secret_key: Optional[str],
        dest: Path,
    ) -> None:
        """Download an S3 URI (tar.gz or zip) and extract it into dest."""
        # Parse S3 URI: s3://bucket-name/path/to/file.tar.gz
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        filename = Path(key).name
        local_file = dest / filename

        # Download
        s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        s3.download_file(bucket, key, str(local_file))

        # Extract
        if filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            with tarfile.open(local_file, "r:gz") as tf:
                tf.extractall(path=dest)
        elif filename.endswith(".zip"):
            with zipfile.ZipFile(local_file, "r") as zf:
                zf.extractall(dest)
        else:
            raise ValueError("S3 file must be .tar.gz, .tgz, or .zip")

    def create_dataloaders(self) -> DataLoaders:
        """Build FastAI DataLoaders based on self.config."""
        data_cfg = self.config["data"]
        train_cfg = self.config["train"]

        # 1) Label function
        fn_src = data_cfg["data_source"]["label_fn"]
        if fn_src.strip().startswith("lambda"):
            label_fn = eval(fn_src)
        else:
            ns = {}
            exec(fn_src, ns)
            label_fn = ns.get("label_fn")
            if not callable(label_fn):
                raise ValueError("label_fn must be a callable")

        # 2) Validation split
        valid_pct = train_cfg.get("validation_split", 0.2)

        # 3) Transforms
        tf = data_cfg.get("transforms", {})
        w, h = tf.get("resize", {"width": 224, "height": 224}).values()
        norm = tf.get(
            "normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        )
        # convert normalize to lists to tensors
        norm["mean"] = tensor(norm["mean"]).float()
        norm["std"] = tensor(norm["std"]).float()
        aug = tf.get("augmentation", {})

        item_tfms = [Resize((h, w))]
        batch_tfms = []

        # Conditional augmentations
        if any([aug.get("flip"), aug.get("rotation"), aug.get("color_jitter")]):
            cj = aug.get("color_jitter", {})
            batch_tfms.append(
                aug_transforms(
                    flip_vert=False,
                    max_rotate=aug.get("rotation", 0),
                    brightness=cj.get("brightness", 0.0),
                    contrast=cj.get("contrast", 0.0),
                    saturation=cj.get("saturation", 0.0),
                    hue=cj.get("hue", 0.0),
                )
            )

        batch_tfms.append(Normalize(mean=norm["mean"], std=norm["std"]))

        # 4) DataBlock
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
            get_y=label_fn,
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
        )

        # 5) DataLoaders
        return dblock.dataloaders(
            source=self.data_path,
            bs=train_cfg.get("batch_size", 32),
            num_workers=train_cfg.get("num_workers", 4),
        )
