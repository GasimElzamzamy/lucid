import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from main import (
    align_labels_to_automata_predictions,
    evaluate_model,
    set_seed,
    train_model,
)
from src.data_pipeline import (
    load_and_prepare_batadal,
    load_config,
    load_skab,
    prepare_skab_cv,
)
from src.models_automata import ProbabilisticAutomata
from src.models_dl import (
    CNN1DAnomalyDetector,
    LSTMAnomalyDetector,
    TimeSeriesDataset,
)

OUTPUT_CSV = os.path.join("results", "cross_dataset_results.csv")
OUTPUT_MD = os.path.join("results", "cross_dataset_results.md")
OUTPUT_JSON = os.path.join("results", "cross_dataset_results.json")


def make_pc1_dataset(X_pc1, y, window_size):
    X_pc1 = np.asarray(X_pc1, dtype=float).reshape(-1, 1)
    return TimeSeriesDataset(X_pc1, y, window_size)


def evaluate_automata_cross_dataset(
    train_pc1,
    test_pc1,
    y_test,
    config,
    window_size=4,
    alphabet_size=3,
):
    model = ProbabilisticAutomata(
        window_size=window_size,
        alphabet_size=alphabet_size,
        laplace_smoothing=config["automata"]["laplace_smoothing"],
    )

    model.fit(train_pc1)

    preds = model.predict(
        test_pc1,
        anomaly_threshold=config["automata"]["anomaly_threshold"],
    )

    y_aligned = align_labels_to_automata_predictions(
        y_test=y_test,
        num_predictions=len(preds),
    )

    min_len = min(len(y_aligned), len(preds))
    y_aligned = y_aligned[:min_len]
    preds = preds[:min_len]

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "f1": f1_score(y_aligned, preds, zero_division=0),
        "precision": precision_score(y_aligned, preds, zero_division=0),
        "recall": recall_score(y_aligned, preds, zero_division=0),
        "accuracy": accuracy_score(y_aligned, preds),
        "num_states": len(model.vocabulary),
        "num_transitions": sum(len(v) for v in model.transition_counts.values()),
    }


def evaluate_dl_cross_dataset(
    model_name,
    train_pc1,
    y_train,
    test_pc1,
    y_test,
    config,
    seed,
):
    set_seed(seed)

    window_size = 5
    batch_size = config["deep_learning"]["batch_size"]

    train_dataset = make_pc1_dataset(train_pc1, y_train, window_size)
    test_dataset = make_pc1_dataset(test_pc1, y_test, window_size)

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

    input_size = 1

    if model_name == "LSTM":
        model = LSTMAnomalyDetector(
            input_size,
            config["deep_learning"]["hidden_units"],
            config["deep_learning"]["dropout_rate"],
        )
    elif model_name == "1D-CNN":
        model = CNN1DAnomalyDetector(
            input_size,
            config["deep_learning"]["hidden_units"],
            config["deep_learning"]["dropout_rate"],
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    trained_model = train_model(model, train_loader, config)

    return evaluate_model(
        trained_model,
        test_loader,
    )


def summarize(rows):
    df = pd.DataFrame(rows)

    summary = (
        df
        .groupby(["train_dataset", "test_dataset", "model"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_accuracy=("accuracy", "mean"),
        )
    )

    return df, summary


def save_markdown(summary_df, output_path):
    lines = []
    lines.append("# Cross-Dataset Generalization Results")
    lines.append("")
    lines.append(
        "All cross-dataset experiments use the one-dimensional PC1 representation "
        "so that BATADAL and SKAB can be compared despite having different raw sensor dimensions."
    )
    lines.append("")
    lines.append("| Train Dataset | Test Dataset | Model | Mean F1 | Std F1 | Mean Precision | Mean Recall | Mean Accuracy |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['train_dataset']} | "
            f"{row['test_dataset']} | "
            f"{row['model']} | "
            f"{row['mean_f1']:.4f} | "
            f"{row['std_f1']:.4f} | "
            f"{row['mean_precision']:.4f} | "
            f"{row['mean_recall']:.4f} | "
            f"{row['mean_accuracy']:.4f} |"
        )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    config = load_config()
    os.makedirs(config["output_dir"], exist_ok=True)

    print("Loading BATADAL...")
    batadal = load_and_prepare_batadal(config)

    print("Loading SKAB...")
    skab_df = load_skab(config)
    skab_splits = prepare_skab_cv(skab_df, config)

    # Cross-dataset için SKAB tarafında ilk GroupKFold splitini temsilci split olarak kullanıyoruz.
    skab = skab_splits[0]

    datasets = {
        "BATADAL": {
            "train_pc1": batadal["X_train_pc1"],
            "y_train": batadal["y_train"],
            "test_pc1": batadal["X_test_pc1"],
            "y_test": batadal["y_test"],
        },
        "SKAB": {
            "train_pc1": skab["X_train_pc1"],
            "y_train": skab["y_train"],
            "test_pc1": skab["X_test_pc1"],
            "y_test": skab["y_test"],
        },
    }

    rows = []

    experiment_pairs = [
        ("BATADAL", "SKAB"),
        ("SKAB", "BATADAL"),
    ]

    for train_name, test_name in experiment_pairs:
        print(f"\nCross-dataset: Train {train_name} -> Test {test_name}")

        train_data = datasets[train_name]
        test_data = datasets[test_name]

        for seed in config["random_seeds"]:
            print(f"  Seed {seed}")

            for model_name in ["LSTM", "1D-CNN"]:
                metrics = evaluate_dl_cross_dataset(
                    model_name=model_name,
                    train_pc1=train_data["train_pc1"],
                    y_train=train_data["y_train"],
                    test_pc1=test_data["test_pc1"],
                    y_test=test_data["y_test"],
                    config=config,
                    seed=seed,
                )

                rows.append({
                    "train_dataset": train_name,
                    "test_dataset": test_name,
                    "seed": seed,
                    "model": model_name,
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "accuracy": None,
                })

                print(f"    {model_name} F1: {metrics['f1']:.4f}")

            automata_metrics = evaluate_automata_cross_dataset(
                train_pc1=train_data["train_pc1"],
                test_pc1=test_data["test_pc1"],
                y_test=test_data["y_test"],
                config=config,
                window_size=4,
                alphabet_size=3,
            )

            rows.append({
                "train_dataset": train_name,
                "test_dataset": test_name,
                "seed": seed,
                "model": "Automata_w4_a3",
                "f1": automata_metrics["f1"],
                "precision": automata_metrics["precision"],
                "recall": automata_metrics["recall"],
                "accuracy": automata_metrics["accuracy"],
            })

            print(f"    Automata_w4_a3 F1: {automata_metrics['f1']:.4f}")

    detailed_df, summary_df = summarize(rows)

    detailed_df.to_csv(OUTPUT_CSV, index=False)
    save_markdown(summary_df, OUTPUT_MD)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=4)

    print(f"\nSaved cross-dataset CSV to {OUTPUT_CSV}")
    print(f"Saved cross-dataset Markdown to {OUTPUT_MD}")
    print(f"Saved cross-dataset JSON to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()