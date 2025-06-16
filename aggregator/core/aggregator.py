from aggregator_service.core.model_utils import get_model, evaluate_model
from aggregator_service.db.database import SessionLocal
from aggregator_service.db import crud, models
from mlflow.models.signature import infer_signature
import mlflow, torch, os

MODEL_DIR = "/mnt/shared"
MODEL_PATH = os.path.join(MODEL_DIR, "global_model.pt")

mlflow.set_tracking_uri("http://fedadmin:fedmed@localhost:8080")
mlflow.set_experiment("FedMedVision")

def aggregate_weights():
    db = SessionLocal()

    # Get current active round
    current_round = db.query(models.Round)\
        .filter_by(status=models.RoundStatus.started)\
        .order_by(models.Round.start_time.desc())\
        .first()

    if not current_round:
        print("=> No active round to aggregate")
        db.close()
        return

    print(f"=> Submitted clients: {current_round.submitted_clients}")
    print(f"=> Selected clients: {current_round.selected_clients}")

    if not current_round.selected_clients or not current_round.submitted_clients:
        print("=> No clients or submissions to aggregate")
        db.close()
        return

    if set(current_round.submitted_clients) != set(current_round.selected_clients):
        print(f"⏳ Waiting for all clients. Received: {len(current_round.submitted_clients)}/{len(current_round.selected_clients)}")
        db.close()
        return

    print(f"🔄 Aggregating for round: {current_round.round_id}")
    files = [f"update_{cid}.pt" for cid in current_round.selected_clients]
    file_paths = [os.path.join(MODEL_DIR, f) for f in files]

    if not all(os.path.exists(fp) for fp in file_paths):
        print("❌ Not all update files are present.")
        db.close()
        return

    updates = [torch.load(fp, map_location="cpu") for fp in file_paths]
    base = updates[0]
    for key in base:
        for i in range(1, len(updates)):
            base[key] += updates[i][key]
        base[key] = base[key] / len(updates)

    torch.save(base, MODEL_PATH)
    print("✅ Aggregated global model saved to:", MODEL_PATH)

    metrics = evaluate_model(MODEL_PATH, round_id=current_round.round_id)

    # Log to MLflow
    with mlflow.start_run(run_name=f"Round_{current_round.round_id}") as run:
        mlflow.log_param("round_id", current_round.round_id)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        mlflow.log_artifact(metrics["confusion_matrix_path"], artifact_path="confusion_matrices")

        model_obj = get_model()
        model_obj.load_state_dict(torch.load(MODEL_PATH))
        model_obj.eval()

        # Log model with example input and signature
        example_input = torch.rand(1, 3, 224, 224)
        example_np = example_input.numpy()
        signature = infer_signature(example_input.numpy(), model_obj(example_input).detach().numpy())

        mlflow.pytorch.log_model(
            model_obj,
            artifact_path="model",
            input_example=example_np,
            signature=signature
        )

        mlflow.register_model(f"runs:/{run.info.run_id}/model", "FedMedGlobalModel")
        print(f"📦 Model registered in MLflow as FedMedGlobalModel@{run.info.run_id}")

    # Store metrics in DB
    crud.update_round_metrics(db, current_round.round_id, metrics)
    db.close()
