import argparse
import pickle
import numpy as np
import torch
import zmq
import time
from fastai.vision.all import *


def is_cat(x):
    "Label function: returns True if the first character of the filename is uppercase (indicating a cat)."
    return x[0].isupper()


def main(args):
    # Set up ZeroMQ context and DEALER socket
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    # Set the DEALER identity based on the node's self number for proper routing
    socket.setsockopt(zmq.IDENTITY, str(args.self_num).encode())
    socket.connect("tcp://localhost:5555")
    print(f"Node {args.self_num}: Connected to aggregator.")

    # Send an initial 'init' message to the aggregator
    init_msg = {"type": "init"}
    socket.send(pickle.dumps(init_msg))
    print(f"Node {args.self_num}: Sent init message to aggregator.")

    # Wait for the initial global model weights from the aggregator
    global_weights_data = socket.recv()
    global_weights = pickle.loads(global_weights_data)
    print(f"Node {args.self_num}: Received global weights from aggregator.")

    # Set up the PETS dataset
    path = untar_data(URLs.PETS) / "images"
    all_files = sorted(get_image_files(path))
    # Split the dataset among the nodes
    parts = np.array_split(all_files, args.num_nodes)
    try:
        my_files = parts[args.self_num].tolist()
    except IndexError:
        raise ValueError("self_num must be between 0 and num_nodes-1")

    print(
        f"Node {args.self_num}: Training on {len(my_files)} images out of {len(all_files)} total images."
    )

    # Create DataLoaders using only this node's subset of data
    dls = ImageDataLoaders.from_name_func(
        path,
        my_files,
        valid_pct=0.2,
        seed=42,
        label_func=is_cat,
        item_tfms=Resize(224),
    )

    # Define the learner using resnet34 and load the initial global weights
    learn = cnn_learner(dls, resnet34, metrics=error_rate)
    learn.model.load_state_dict(global_weights)

    # Define the number of training rounds to simulate federated updates
    num_rounds = 3
    for rnd in range(num_rounds):
        print(f"Node {args.self_num}: Starting training round {rnd+1}...")
        # Perform local training (here, fine_tune for one epoch)
        learn.fine_tune(1)

        # After training, send the updated local weights to the aggregator
        local_weights = learn.model.state_dict()
        msg = {"type": "weights", "weights": local_weights}
        socket.send(pickle.dumps(msg))
        print(f"Node {args.self_num}: Sent local weights to aggregator.")

        # Wait for aggregated weights from the aggregator
        aggregated_weights_data = socket.recv()
        aggregated_weights = pickle.loads(aggregated_weights_data)
        print(f"Node {args.self_num}: Received aggregated weights from aggregator.")

        # Update the local model with the aggregated global weights
        learn.model.load_state_dict(aggregated_weights)
        print(f"Node {args.self_num}: Updated local model with aggregated weights.")

    # Save the final model locally
    model_filename = f"model_node_{args.self_num}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(learn.model, f)
    print(f"Node {args.self_num}: Final model saved as {model_filename}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Distributed Training Node")
#     parser.add_argument(
#         "--num_nodes", type=int, required=True, help="Total number of nodes"
#     )
#     parser.add_argument(
#         "--self_num", type=int, required=True, help="ID of the current node (0-indexed)"
#     )
#     args = parser.parse_args()
#     main(args)


# test_mnist_dataset_loader.py

import json
from pathlib import Path
from fastai.data.external import URLs, untar_data
from fastai.vision.all import accuracy
from data import Dataset

# 1. JSON config for MNIST_SAMPLE
config_json = """
{
  "task": "mnist_test",
  "task_type": "classification",
  "data": {
    "data_source": {
      "type": "url",
      "url": "%(mnist_url)s",
      "label_fn": "def label_fn(x): from pathlib import Path; return Path(x).parent.name"
    },
    "classes": ["0","1","2","3","4","5","6","7","8","9"],
    "transforms": {
      "resize": {"width": 28, "height": 28},
      "normalize": {"mean": [0.5], "std": [0.5]},
      "augmentation": {"flip": false, "rotation": 0}
    }
  },
  "train": {
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 1,
    "validation_split": 0.2
  }
}
""" % {
    "mnist_url": URLs.MNIST_SAMPLE
}

# 2. Load config
config = json.loads(config_json)

# 3. Instantiate Dataset and create dataloaders
dataset = Dataset(config)
dls = dataset.create_dataloaders()

# 4. Quick sanity checks
print("— Dataloaders —")
print(f"  Train batches per epoch: {len(dls.train)}")
print(f"  Valid batches per epoch: {len(dls.valid)}")

# Grab one batch and inspect shapes
x, y = dls.one_batch()
print("\n— One batch shapes —")
print(f"  x: {x.shape}   (bs, channels, height, width)")
print(f"  y: {y.shape}   (bs,)         labels")

# 5. (Optional) train a single batch through a learner
from fastai.vision.all import vision_learner, resnet18

learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fit_one_cycle(1, 1e-3)
# loss_before = learn.loss_func(learn.model(x), y)
print(f"\nInitial loss on one batch: {loss_before}")
