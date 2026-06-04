import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

from src.data_pipeline import load_config, load_skab, prepare_skab_cv, load_and_prepare_batadal
from src.models_dl import TimeSeriesDataset, LSTMAnomalyDetector, CNN1DAnomalyDetector
from src.models_automata import ProbabilisticAutomata
from src.explainer import AutomataExplainer
from src.visualizations import plot_confusion_matrix, plot_roc_curve, plot_pr_curve


def set_seed(seed):
    """Locks all random number generators for strict reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_gaussian_noise(X, config):
    """Injects noise to test model robustness if enabled in config."""
    noise_cfg = config["scenarios"]["gaussian_noise"]
    if noise_cfg["apply"]:
        noise = np.random.normal(noise_cfg["mean"], noise_cfg["std_dev"], X.shape)
        return X + noise
    return X


def train_model(model, train_loader, config):
    """Standard PyTorch training loop."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["deep_learning"]["learning_rate"]
    )
    epochs = config["deep_learning"]["epochs"]

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, test_loader, apply_noise=False, config=None, return_arrays=False):
    """Evaluates a deep learning model."""
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if apply_noise and config:
                X_np = X_batch.numpy()
                X_noisy = apply_gaussian_noise(X_np, config)
                X_batch = torch.tensor(X_noisy, dtype=torch.float32)

            predictions = model(X_batch)
            binary_preds = (predictions > 0.5).float()

        all_probs.extend(np.atleast_1d(predictions.detach().cpu().numpy()).reshape(-1))
        all_preds.extend(np.atleast_1d(binary_preds.detach().cpu().numpy()).reshape(-1))
        all_targets.extend(np.atleast_1d(y_batch.detach().cpu().numpy()).reshape(-1))

    result = {
        "f1": f1_score(all_targets, all_preds, zero_division=0),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0)
    }

    if return_arrays:
        result["targets"] = all_targets
        result["preds"] = all_preds
        result["probs"] = all_probs

    return result


def align_labels_to_automata_predictions(y_test, num_predictions):
    """
    Automata predictions are produced on SAX/pattern level, not raw row level.
    If any raw label inside a segment is anomalous, the segment label becomes anomaly.
    """
    y_test = np.asarray(y_test, dtype=int)

    if num_predictions <= 0:
        return np.array([], dtype=int)

    segments = np.array_split(y_test, num_predictions)

    return np.array(
        [1 if np.any(segment == 1) else 0 for segment in segments],
        dtype=int
    )


def evaluate_automata(
    X_train_pc1,
    X_test_pc1,
    y_test,
    config,
    window_size,
    alphabet_size,
    apply_noise=False
):
    """
    Trains and evaluates the Probabilistic Automata model on PC1 series.
    Labels are aligned from raw row-level labels to SAX/pattern-level labels.
    """
    if apply_noise:
        X_test_pc1 = apply_gaussian_noise(X_test_pc1, config)

    automata = ProbabilisticAutomata(
        window_size=window_size,
        alphabet_size=alphabet_size,
        laplace_smoothing=config["automata"]["laplace_smoothing"]
    )

    automata.fit(X_train_pc1)

    explainer = AutomataExplainer(
        config["automata"]["anomaly_threshold"]
    )

    preds = automata.predict(
        X_test_pc1,
        anomaly_threshold=config["automata"]["anomaly_threshold"]
    )

    y_aligned = align_labels_to_automata_predictions(
        y_test=y_test,
        num_predictions=len(preds)
    )

    min_len = min(len(y_aligned), len(preds))
    y_aligned = y_aligned[:min_len]
    preds = preds[:min_len]

    return {
        "accuracy": accuracy_score(y_aligned, preds),
        "f1": f1_score(y_aligned, preds, zero_division=0),
        "precision": precision_score(y_aligned, preds, zero_division=0),
        "recall": recall_score(y_aligned, preds, zero_division=0),
        "num_states": len(automata.vocabulary),
        "num_transitions": sum(len(v) for v in automata.transition_counts.values()),
        "sample_explanations": [
            json.loads(
                explainer.to_json(explanation)
            )
            for explanation in automata.last_explanations[:5]
        ]
    }


def main():
    config = load_config()
    os.makedirs(config["output_dir"], exist_ok=True)

    window_size = 5
    automata_window_size = 5
    automata_alphabet_size = 4
    batch_size = config["deep_learning"]["batch_size"]

    print("Loading datasets...")
    batadal_data = load_and_prepare_batadal(config)

    results_log = {
        "BATADAL": {},
        "SKAB": {}
    }

    for seed in config["random_seeds"]:
        print(f"\n--- Running Experiment for Seed: {seed} ---")
        set_seed(seed)

        # ==========================
        # BATADAL EXPERIMENT
        # ==========================
        print("Training on BATADAL...")

        train_dataset = TimeSeriesDataset(
            batadal_data["X_train"],
            batadal_data["y_train"],
            window_size
        )
        test_dataset = TimeSeriesDataset(
            batadal_data["X_test"],
            batadal_data["y_test"],
            window_size
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = batadal_data["X_train"].shape[1]

        models = {
            "LSTM": LSTMAnomalyDetector(
                input_size,
                config["deep_learning"]["hidden_units"],
                config["deep_learning"]["dropout_rate"]
            ),
            "1D-CNN": CNN1DAnomalyDetector(
                input_size,
                config["deep_learning"]["hidden_units"],
                config["deep_learning"]["dropout_rate"]
            )
        }

        results_log["BATADAL"][seed] = {}

        for model_name, model in models.items():
            print(f"  -> Training {model_name}...")

            trained_model = train_model(model, train_loader, config)
            needs_plots = seed == 42

            metrics_clean = evaluate_model(
                trained_model,
                test_loader,
                return_arrays=needs_plots
            )

            if needs_plots:
                print(f"     -> Generating plots for BATADAL {model_name}...")

                plot_confusion_matrix(
                    metrics_clean["targets"],
                    metrics_clean["preds"],
                    f"BATADAL {model_name} Confusion Matrix",
                    os.path.join(config["output_dir"], f"BATADAL_{model_name}_CM.png")
                )

                plot_roc_curve(
                    metrics_clean["targets"],
                    metrics_clean["probs"],
                    f"BATADAL {model_name} ROC Curve",
                    os.path.join(config["output_dir"], f"BATADAL_{model_name}_ROC.png")
                )

            metrics_noisy = evaluate_model(
                trained_model,
                test_loader,
                apply_noise=True,
                config=config
            )

            results_log["BATADAL"][seed][model_name] = {
                "clean": {
                    "f1": metrics_clean["f1"],
                    "precision": metrics_clean["precision"],
                    "recall": metrics_clean["recall"]
                },
                "noisy": {
                    "f1": metrics_noisy["f1"],
                    "precision": metrics_noisy["precision"],
                    "recall": metrics_noisy["recall"]
                }
            }

            print(
                f"     {model_name} Clean F1: {metrics_clean['f1']:.4f} | "
                f"Noisy F1: {metrics_noisy['f1']:.4f}"
            )

        print("  -> Training Automata on BATADAL...")

        automata_clean = evaluate_automata(
            X_train_pc1=batadal_data["X_train_pc1"],
            X_test_pc1=batadal_data["X_test_pc1"],
            y_test=batadal_data["y_test"],
            config=config,
            window_size=automata_window_size,
            alphabet_size=automata_alphabet_size,
            apply_noise=False
        )

        automata_noisy = evaluate_automata(
            X_train_pc1=batadal_data["X_train_pc1"],
            X_test_pc1=batadal_data["X_test_pc1"],
            y_test=batadal_data["y_test"],
            config=config,
            window_size=automata_window_size,
            alphabet_size=automata_alphabet_size,
            apply_noise=True
        )

        results_log["BATADAL"][seed]["Automata"] = {
            "clean": automata_clean,
            "noisy": automata_noisy
        }

        print(
            f"     Automata Clean F1: {automata_clean['f1']:.4f} | "
            f"Noisy F1: {automata_noisy['f1']:.4f}"
        )

        # ==========================
        # SKAB EXPERIMENT
        # ==========================
        print("Training on SKAB (GroupKFold)...")

        skab_df = load_skab(config)
        skab_splits = prepare_skab_cv(skab_df, config)

        fold_metrics = {
            "LSTM": {
                "clean": {"f1": [], "precision": [], "recall": []},
                "noisy": {"f1": [], "precision": [], "recall": []}
            },
            "1D-CNN": {
                "clean": {"f1": [], "precision": [], "recall": []},
                "noisy": {"f1": [], "precision": [], "recall": []}
            },
            "Automata": {
                "clean": {"f1": [], "precision": [], "recall": [], "accuracy": []},
                "noisy": {"f1": [], "precision": [], "recall": [], "accuracy": []}
            }
        }

        for fold_idx, fold_data in enumerate(skab_splits):
            print(f"  -> SKAB Fold {fold_idx + 1}/{len(skab_splits)}")

            train_dataset_skab = TimeSeriesDataset(
                fold_data["X_train"],
                fold_data["y_train"],
                window_size
            )
            test_dataset_skab = TimeSeriesDataset(
                fold_data["X_test"],
                fold_data["y_test"],
                window_size
            )

            train_loader_skab = DataLoader(
                train_dataset_skab,
                batch_size=batch_size,
                shuffle=True
            )
            test_loader_skab = DataLoader(
                test_dataset_skab,
                batch_size=batch_size,
                shuffle=False
            )

            input_size_skab = fold_data["X_train"].shape[1]

            models_skab = {
                "LSTM": LSTMAnomalyDetector(
                    input_size_skab,
                    config["deep_learning"]["hidden_units"],
                    config["deep_learning"]["dropout_rate"]
                ),
                "1D-CNN": CNN1DAnomalyDetector(
                    input_size_skab,
                    config["deep_learning"]["hidden_units"],
                    config["deep_learning"]["dropout_rate"]
                )
            }

            for model_name, model in models_skab.items():
                trained_model = train_model(model, train_loader_skab, config)

                needs_plots = seed == 42 and fold_idx == 0

                metrics_clean = evaluate_model(
                    trained_model,
                    test_loader_skab,
                    return_arrays=needs_plots
                )

                if needs_plots:
                    print(f"     -> Generating plots for SKAB {model_name}...")

                    plot_confusion_matrix(
                        metrics_clean["targets"],
                        metrics_clean["preds"],
                        f"SKAB {model_name} Confusion Matrix",
                        os.path.join(config["output_dir"], f"SKAB_{model_name}_CM.png")
                    )

                    plot_roc_curve(
                        metrics_clean["targets"],
                        metrics_clean["probs"],
                        f"SKAB {model_name} ROC Curve",
                        os.path.join(config["output_dir"], f"SKAB_{model_name}_ROC.png")
                    )

                fold_metrics[model_name]["clean"]["f1"].append(metrics_clean["f1"])
                fold_metrics[model_name]["clean"]["precision"].append(metrics_clean["precision"])
                fold_metrics[model_name]["clean"]["recall"].append(metrics_clean["recall"])

                metrics_noisy = evaluate_model(
                    trained_model,
                    test_loader_skab,
                    apply_noise=True,
                    config=config
                )

                fold_metrics[model_name]["noisy"]["f1"].append(metrics_noisy["f1"])
                fold_metrics[model_name]["noisy"]["precision"].append(metrics_noisy["precision"])
                fold_metrics[model_name]["noisy"]["recall"].append(metrics_noisy["recall"])

            print("     -> Training Automata on this SKAB fold...")

            skab_automata_clean = evaluate_automata(
                X_train_pc1=fold_data["X_train_pc1"],
                X_test_pc1=fold_data["X_test_pc1"],
                y_test=fold_data["y_test"],
                config=config,
                window_size=automata_window_size,
                alphabet_size=automata_alphabet_size,
                apply_noise=False
            )

            skab_automata_noisy = evaluate_automata(
                X_train_pc1=fold_data["X_train_pc1"],
                X_test_pc1=fold_data["X_test_pc1"],
                y_test=fold_data["y_test"],
                config=config,
                window_size=automata_window_size,
                alphabet_size=automata_alphabet_size,
                apply_noise=True
            )

            for metric_name in ["f1", "precision", "recall", "accuracy"]:
                fold_metrics["Automata"]["clean"][metric_name].append(
                    skab_automata_clean[metric_name]
                )
                fold_metrics["Automata"]["noisy"][metric_name].append(
                    skab_automata_noisy[metric_name]
                )

            print(
                f"        Automata Clean Fold F1: {skab_automata_clean['f1']:.4f} | "
                f"Noisy Fold F1: {skab_automata_noisy['f1']:.4f}"
            )

        results_log["SKAB"][seed] = {}

        for model_name in ["LSTM", "1D-CNN"]:
            results_log["SKAB"][seed][model_name] = {
                "clean": {},
                "noisy": {}
            }

            for condition in ["clean", "noisy"]:
                results_log["SKAB"][seed][model_name][condition] = {
                    "f1": float(np.mean(fold_metrics[model_name][condition]["f1"])),
                    "precision": float(np.mean(fold_metrics[model_name][condition]["precision"])),
                    "recall": float(np.mean(fold_metrics[model_name][condition]["recall"])),
                    "f1_std": float(np.std(fold_metrics[model_name][condition]["f1"]))
                }

            print(
                f"     SKAB {model_name} Clean F1: "
                f"{results_log['SKAB'][seed][model_name]['clean']['f1']:.4f} | "
                f"Noisy F1: "
                f"{results_log['SKAB'][seed][model_name]['noisy']['f1']:.4f}"
            )

        results_log["SKAB"][seed]["Automata"] = {
            "clean": {},
            "noisy": {}
        }

        for condition in ["clean", "noisy"]:
            results_log["SKAB"][seed]["Automata"][condition] = {
                "f1": float(np.mean(fold_metrics["Automata"][condition]["f1"])),
                "precision": float(np.mean(fold_metrics["Automata"][condition]["precision"])),
                "recall": float(np.mean(fold_metrics["Automata"][condition]["recall"])),
                "accuracy": float(np.mean(fold_metrics["Automata"][condition]["accuracy"])),
                "f1_std": float(np.std(fold_metrics["Automata"][condition]["f1"]))
            }

        print(
            f"     SKAB Automata Clean F1: "
            f"{results_log['SKAB'][seed]['Automata']['clean']['f1']:.4f} | "
            f"Noisy F1: "
            f"{results_log['SKAB'][seed]['Automata']['noisy']['f1']:.4f}"
        )

    results_path = os.path.join(config["output_dir"], "dl_results.json")

    with open(results_path, "w") as f:
        json.dump(results_log, f, indent=4)

    print(f"\n✅ All runs complete! Results saved to {results_path}")


if __name__ == "__main__":
    main()