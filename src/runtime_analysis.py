import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.data_pipeline import load_config, load_and_prepare_batadal
from src.models_automata import ProbabilisticAutomata
from src.models_dl import (
    CNN1DAnomalyDetector,
    LSTMAnomalyDetector,
    TimeSeriesDataset,
)
from main import train_model, evaluate_model, evaluate_automata


def measure_deep_learning_runtime(model_name, model, train_loader, test_loader, config):
    start_train = time.perf_counter()
    trained_model = train_model(model, train_loader, config)
    end_train = time.perf_counter()

    start_inference = time.perf_counter()
    _ = evaluate_model(trained_model, test_loader)
    end_inference = time.perf_counter()

    return {
        "model": model_name,
        "training_time_seconds": end_train - start_train,
        "inference_time_seconds": end_inference - start_inference,
    }


def measure_automata_runtime(data, config):
    start_train = time.perf_counter()

    model = ProbabilisticAutomata(
        window_size=5,
        alphabet_size=3,
        laplace_smoothing=config["automata"]["laplace_smoothing"],
    )

    model.fit(data["X_train_pc1"])

    end_train = time.perf_counter()

    start_inference = time.perf_counter()

    _ = model.predict(
        data["X_test_pc1"],
        anomaly_threshold=config["automata"]["anomaly_threshold"],
    )

    end_inference = time.perf_counter()

    return {
        "model": "Automata",
        "training_time_seconds": end_train - start_train,
        "inference_time_seconds": end_inference - start_inference,
    }


def save_markdown(rows, output_path):
    lines = []
    lines.append("# Runtime Analysis")
    lines.append("")
    lines.append("| Model | Training Time (s) | Inference Time (s) |")
    lines.append("|---|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['model']} | "
            f"{row['training_time_seconds']:.6f} | "
            f"{row['inference_time_seconds']:.6f} |"
        )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    config = load_config()
    os.makedirs(config["output_dir"], exist_ok=True)

    data = load_and_prepare_batadal(config)

    window_size = 5
    batch_size = config["deep_learning"]["batch_size"]

    train_dataset = TimeSeriesDataset(
        data["X_train"],
        data["y_train"],
        window_size,
    )

    test_dataset = TimeSeriesDataset(
        data["X_test"],
        data["y_test"],
        window_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    input_size = data["X_train"].shape[1]

    rows = []

    rows.append(
        measure_deep_learning_runtime(
            model_name="LSTM",
            model=LSTMAnomalyDetector(
                input_size,
                config["deep_learning"]["hidden_units"],
                config["deep_learning"]["dropout_rate"],
            ),
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
        )
    )

    rows.append(
        measure_deep_learning_runtime(
            model_name="1D-CNN",
            model=CNN1DAnomalyDetector(
                input_size,
                config["deep_learning"]["hidden_units"],
                config["deep_learning"]["dropout_rate"],
            ),
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
        )
    )

    rows.append(
        measure_automata_runtime(
            data=data,
            config=config,
        )
    )

    json_path = os.path.join(config["output_dir"], "runtime_analysis.json")
    md_path = os.path.join(config["output_dir"], "runtime_analysis.md")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=4)

    save_markdown(rows, md_path)

    print(f"Saved runtime JSON to {json_path}")
    print(f"Saved runtime Markdown to {md_path}")


if __name__ == "__main__":
    main()