import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, accuracy_score
from torch.utils.data import DataLoader

from src.data_pipeline import (
    load_and_prepare_batadal,
    load_config,
    load_skab,
    prepare_skab_cv,
)
from src.explainer import AutomataExplainer
from src.models_automata import ProbabilisticAutomata
from src.models_dl import CNN1DAnomalyDetector, LSTMAnomalyDetector, TimeSeriesDataset
from src.visualizations import plot_confusion_matrix, plot_roc_curve, plot_pr_curve

# ==========================================
# 1. UTILITIES & REPRODUCIBILITY
# ==========================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def apply_gaussian_noise(X, config):
    noise_cfg = config["scenarios"]["gaussian_noise"]
    if noise_cfg["apply"]:
        noise = np.random.normal(noise_cfg["mean"], noise_cfg["std_dev"], X.shape)
        return X + noise
    return X

# ==========================================
# 2. TRAINING & EVALUATION ENGINES
# ==========================================
def _calculate_positive_weight(dataset):
    y = dataset.y[dataset.window_size - 1:]
    positives = torch.sum(y == 1).item()
    negatives = torch.sum(y == 0).item()
    if positives == 0:
        return 1.0
    return max(1.0, negatives / positives)

def _evaluate_validation_loss(model, val_loader, criterion, positive_weight):
    model.eval()
    losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            raw_logits = model(X_batch)
            loss_values = criterion(raw_logits, y_batch)
            sample_weights = torch.where(
                y_batch == 1,
                torch.tensor(positive_weight, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            )
            loss = (loss_values * sample_weights).mean()
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")

def train_model(model, train_loader, config, val_loader=None):
    """Trains a DL model with class-weighted BCE loss and early stopping."""
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["deep_learning"]["learning_rate"],
    )

    positive_weight = _calculate_positive_weight(train_loader.dataset)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = config["deep_learning"].get("early_stopping_patience", 5)

    model.train()
    for _ in range(config["deep_learning"]["epochs"]):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            raw_logits = model(X_batch)
            loss_values = criterion(raw_logits, y_batch)

            sample_weights = torch.where(
                y_batch == 1,
                torch.tensor(positive_weight, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            )

            loss = (loss_values * sample_weights).mean()
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            val_loss = _evaluate_validation_loss(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                positive_weight=positive_weight,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def evaluate_model(model, test_loader, apply_noise=False, config=None, return_arrays=False):
    """Evaluates DL models using Dynamic Thresholding."""
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if apply_noise and config:
                X_np = X_batch.numpy()
                X_noisy = apply_gaussian_noise(X_np, config)
                X_batch = torch.tensor(X_noisy, dtype=torch.float32)

            raw_logits = model(X_batch)
            probs = torch.sigmoid(raw_logits) 
            
            all_probs.extend(probs.numpy())
            all_targets.extend(y_batch.numpy())
            
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
            
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    if len(thresholds) > 0:
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5

    binary_preds = (all_probs >= best_threshold).astype(int)
    
    result = {
        "accuracy": accuracy_score(all_targets, binary_preds),
        "f1": f1_score(all_targets, binary_preds, zero_division=0),
        "precision": precision_score(all_targets, binary_preds, zero_division=0),
        "recall": recall_score(all_targets, binary_preds, zero_division=0),
        "threshold": float(best_threshold)
    }
    
    if return_arrays:
        result["targets"] = all_targets
        result["preds"] = binary_preds
        result["probs"] = all_probs

    return result

# ==========================================
# 3. AUTOMATA EVALUATION ENGINES
# ==========================================
def align_labels_to_automata_predictions(y_test, num_predictions):
    y_test = np.asarray(y_test, dtype=int)
    if num_predictions <= 0:
        return np.array([], dtype=int)
    segments = np.array_split(y_test, num_predictions)
    return np.array([1 if np.any(segment == 1) else 0 for segment in segments], dtype=int)

def evaluate_automata(X_train_pc1, X_test_pc1, y_test, config, window_size, alphabet_size, apply_noise=False):
    if apply_noise:
        X_test_pc1 = apply_gaussian_noise(X_test_pc1, config)

    automata = ProbabilisticAutomata(
        window_size=window_size,
        alphabet_size=alphabet_size,
        laplace_smoothing=config["automata"]["laplace_smoothing"],
    )

    automata.fit(X_train_pc1)
    explainer = AutomataExplainer(config["automata"]["anomaly_threshold"])

    preds = automata.predict(
        X_test_pc1,
        anomaly_threshold=config["automata"]["anomaly_threshold"],
    )

    y_aligned = align_labels_to_automata_predictions(y_test=y_test, num_predictions=len(preds))
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
            json.loads(explainer.to_json(explanation))
            for explanation in automata.last_explanations[:5]
        ],
    }

def run_automata_grid_for_dataset(X_train_pc1, X_test_pc1, y_test, config):
    results = {}
    for window_size in config["automata"]["window_sizes"]:
        for alphabet_size in config["automata"]["alphabet_sizes"]:
            experiment_name = f"w{window_size}_a{alphabet_size}"
            clean = evaluate_automata(X_train_pc1, X_test_pc1, y_test, config, window_size, alphabet_size, apply_noise=False)
            noisy = evaluate_automata(X_train_pc1, X_test_pc1, y_test, config, window_size, alphabet_size, apply_noise=True)
            results[experiment_name] = {"clean": clean, "noisy": noisy}
            print(f"     Automata {experiment_name} Clean F1: {clean['f1']:.4f} | Noisy F1: {noisy['f1']:.4f}")
    return results

def summarize_skab_automata_grid(fold_metrics):
    summary = {}
    for experiment_name, experiment_data in fold_metrics.items():
        summary[experiment_name] = {"clean": {}, "noisy": {}}
        for condition in ["clean", "noisy"]:
            summary[experiment_name][condition] = {
                "f1": float(np.mean(experiment_data[condition]["f1"])),
                "precision": float(np.mean(experiment_data[condition]["precision"])),
                "recall": float(np.mean(experiment_data[condition]["recall"])),
                "accuracy": float(np.mean(experiment_data[condition]["accuracy"])),
                "f1_std": float(np.std(experiment_data[condition]["f1"])),
            }
    return summary

# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    config = load_config()
    os.makedirs(config["output_dir"], exist_ok=True)

    dl_window_size = config.get("dl_window_size", 5) 
    batch_size = config["deep_learning"]["batch_size"]

    print("Loading datasets...")
    batadal_data = load_and_prepare_batadal(config)

    results_log = {"BATADAL": {}, "SKAB": {}}

    for seed in config["random_seeds"]:
        print(f"\n--- Running Experiment for Seed: {seed} ---")
        set_seed(seed)

        # ------------------------------------------
        # BATADAL EXECUTION
        # ------------------------------------------
        print("Training on BATADAL...")
        train_dataset = TimeSeriesDataset(batadal_data["X_train"], batadal_data["y_train"], dl_window_size)
        val_dataset = TimeSeriesDataset(batadal_data["X_val"], batadal_data["y_val"], dl_window_size)
        test_dataset = TimeSeriesDataset(batadal_data["X_test"], batadal_data["y_test"], dl_window_size)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = batadal_data["X_train"].shape[1]

        models = {
            "LSTM": LSTMAnomalyDetector(input_size, config["deep_learning"]["hidden_units"], config["deep_learning"]["dropout_rate"]),
            "1D-CNN": CNN1DAnomalyDetector(input_size, config["deep_learning"]["hidden_units"], config["deep_learning"]["dropout_rate"]),
        }

        results_log["BATADAL"][seed] = {}

        # DL Loop
        for model_name, model in models.items():
            print(f"  -> Training {model_name}...")
            trained_model = train_model(model, train_loader, config, val_loader=val_loader)
            needs_plots = seed == config["random_seeds"][0]

            clean_metrics = evaluate_model(trained_model, test_loader, return_arrays=needs_plots)

            if needs_plots:
                print(f"     -> Generating plots for BATADAL {model_name}...")
                plot_confusion_matrix(clean_metrics["targets"], clean_metrics["preds"], f"BATADAL {model_name} Confusion Matrix", os.path.join(config["output_dir"], f"BATADAL_{model_name}_CM.png"))
                plot_roc_curve(clean_metrics["targets"], clean_metrics["probs"], f"BATADAL {model_name} ROC Curve", os.path.join(config["output_dir"], f"BATADAL_{model_name}_ROC.png"))
                plot_pr_curve(clean_metrics["targets"], clean_metrics["probs"], f"BATADAL {model_name} Precision-Recall Curve", os.path.join(config["output_dir"], f"BATADAL_{model_name}_PR.png"))

            noisy_metrics = evaluate_model(trained_model, test_loader, apply_noise=True, config=config)

            results_log["BATADAL"][seed][model_name] = {
                "clean": {"accuracy": clean_metrics["accuracy"], "f1": clean_metrics["f1"], "precision": clean_metrics["precision"], "recall": clean_metrics["recall"]},
                "noisy": {"accuracy": noisy_metrics["accuracy"], "f1": noisy_metrics["f1"], "precision": noisy_metrics["precision"], "recall": noisy_metrics["recall"]},
            }
            print(f"     {model_name} Clean F1: {clean_metrics['f1']:.4f} | Noisy F1: {noisy_metrics['f1']:.4f}")

        # Automata Loop
        print("  -> Training Automata grid on BATADAL...")
        results_log["BATADAL"][seed]["Automata"] = run_automata_grid_for_dataset(
            X_train_pc1=batadal_data["X_train_pc1"],
            X_test_pc1=batadal_data["X_test_pc1"],
            y_test=batadal_data["y_test"],
            config=config,
        )

        # ------------------------------------------
        # SKAB EXECUTION
        # ------------------------------------------
        print("Training on SKAB (GroupKFold)...")
        skab_df = load_skab(config)
        skab_splits = prepare_skab_cv(skab_df, config)

        fold_metrics = {
            "LSTM": {"clean": {"f1": [], "precision": [], "recall": [], "accuracy": []}, "noisy": {"f1": [], "precision": [], "recall": [], "accuracy": []}},
            "1D-CNN": {"clean": {"f1": [], "precision": [], "recall": [], "accuracy": []}, "noisy": {"f1": [], "precision": [], "recall": [], "accuracy": []}},
            "Automata": {},
        }

        for fold_idx, fold_data in enumerate(skab_splits):
            print(f"  -> SKAB Fold {fold_idx + 1}/{len(skab_splits)}")
            train_dataset_skab = TimeSeriesDataset(fold_data["X_train"], fold_data["y_train"], dl_window_size)
            test_dataset_skab = TimeSeriesDataset(fold_data["X_test"], fold_data["y_test"], dl_window_size)

            train_loader_skab = DataLoader(train_dataset_skab, batch_size=batch_size, shuffle=True)
            test_loader_skab = DataLoader(test_dataset_skab, batch_size=batch_size, shuffle=False)

            input_size_skab = fold_data["X_train"].shape[1]

            models_skab = {
                "LSTM": LSTMAnomalyDetector(input_size_skab, config["deep_learning"]["hidden_units"], config["deep_learning"]["dropout_rate"]),
                "1D-CNN": CNN1DAnomalyDetector(input_size_skab, config["deep_learning"]["hidden_units"], config["deep_learning"]["dropout_rate"]),
            }

            # SKAB DL Loop
            for model_name, model in models_skab.items():
                trained_model = train_model(model, train_loader_skab, config)
                needs_plots = seed == config["random_seeds"][0] and fold_idx == 0

                clean_metrics = evaluate_model(trained_model, test_loader_skab, return_arrays=needs_plots)

                if needs_plots:
                    print(f"     -> Generating plots for SKAB {model_name}...")
                    plot_confusion_matrix(clean_metrics["targets"], clean_metrics["preds"], f"SKAB {model_name} Confusion Matrix", os.path.join(config["output_dir"], f"SKAB_{model_name}_CM.png"))
                    plot_roc_curve(clean_metrics["targets"], clean_metrics["probs"], f"SKAB {model_name} ROC Curve", os.path.join(config["output_dir"], f"SKAB_{model_name}_ROC.png"))

                noisy_metrics = evaluate_model(trained_model, test_loader_skab, apply_noise=True, config=config)

                for metric_name in ["f1", "precision", "recall", "accuracy"]:
                    fold_metrics[model_name]["clean"][metric_name].append(clean_metrics[metric_name])
                    fold_metrics[model_name]["noisy"][metric_name].append(noisy_metrics[metric_name])

            # SKAB Automata Loop
            print("     -> Training Automata grid on this SKAB fold...")
            for window_size in config["automata"]["window_sizes"]:
                for alphabet_size in config["automata"]["alphabet_sizes"]:
                    experiment_name = f"w{window_size}_a{alphabet_size}"
                    if experiment_name not in fold_metrics["Automata"]:
                        fold_metrics["Automata"][experiment_name] = {"clean": {"f1": [], "precision": [], "recall": [], "accuracy": []}, "noisy": {"f1": [], "precision": [], "recall": [], "accuracy": []}}

                    clean_automata = evaluate_automata(fold_data["X_train_pc1"], fold_data["X_test_pc1"], fold_data["y_test"], config, window_size, alphabet_size, apply_noise=False)
                    noisy_automata = evaluate_automata(fold_data["X_train_pc1"], fold_data["X_test_pc1"], fold_data["y_test"], config, window_size, alphabet_size, apply_noise=True)

                    for metric_name in ["f1", "precision", "recall", "accuracy"]:
                        fold_metrics["Automata"][experiment_name]["clean"][metric_name].append(clean_automata[metric_name])
                        fold_metrics["Automata"][experiment_name]["noisy"][metric_name].append(noisy_automata[metric_name])

                    print(f"        Automata {experiment_name} Clean Fold F1: {clean_automata['f1']:.4f} | Noisy Fold F1: {noisy_automata['f1']:.4f}")

        # Average SKAB Results
        results_log["SKAB"][seed] = {}
        for model_name in ["LSTM", "1D-CNN"]:
            results_log["SKAB"][seed][model_name] = {"clean": {}, "noisy": {}}
            for condition in ["clean", "noisy"]:
                results_log["SKAB"][seed][model_name][condition] = {
                    "accuracy": float(np.mean(fold_metrics[model_name][condition]["accuracy"])),
                    "f1": float(np.mean(fold_metrics[model_name][condition]["f1"])),
                    "precision": float(np.mean(fold_metrics[model_name][condition]["precision"])),
                    "recall": float(np.mean(fold_metrics[model_name][condition]["recall"])),
                    "f1_std": float(np.std(fold_metrics[model_name][condition]["f1"])),
                }
            print(f"     SKAB {model_name} Clean F1: {results_log['SKAB'][seed][model_name]['clean']['f1']:.4f} | Noisy F1: {results_log['SKAB'][seed][model_name]['noisy']['f1']:.4f}")

        results_log["SKAB"][seed]["Automata"] = summarize_skab_automata_grid(fold_metrics["Automata"])
        for experiment_name, experiment_data in results_log["SKAB"][seed]["Automata"].items():
            print(f"     SKAB Automata {experiment_name} Clean F1: {experiment_data['clean']['f1']:.4f} | Noisy F1: {experiment_data['noisy']['f1']:.4f}")

        # ------------------------------------------
        # CROSS-DATASET GENERALIZATION
        # ------------------------------------------
        if config['data'].get('apply_pca', False) and config['data'].get('pca_components', 1) == 1:
            print("\n  -> Running Cross-Dataset Generalization (PCA=1)...")
            results_log.setdefault("CROSS_DATASET", {})
            results_log["CROSS_DATASET"][seed] = {"Train_BATADAL_Test_SKAB": {}, "Train_SKAB_Test_BATADAL": {}}
            
            print("     [1/2] Training on BATADAL, Testing on SKAB...")
            cross_models_bat = {
                "LSTM": LSTMAnomalyDetector(1, config['deep_learning']['hidden_units'], config['deep_learning']['dropout_rate']),
                "1D-CNN": CNN1DAnomalyDetector(1, config['deep_learning']['hidden_units'], config['deep_learning']['dropout_rate'])
            }
            
            test_dataset_skab_cross = TimeSeriesDataset(skab_splits[0]['X_test'], skab_splits[0]['y_test'], dl_window_size)
            test_loader_skab_cross = DataLoader(test_dataset_skab_cross, batch_size=batch_size, shuffle=False)
            
            for model_name, model in cross_models_bat.items():
                trained_cross = train_model(model, train_loader, config) 
                metrics = evaluate_model(trained_cross, test_loader_skab_cross)
                results_log["CROSS_DATASET"][seed]["Train_BATADAL_Test_SKAB"][model_name] = metrics
                print(f"        [{model_name}] BATADAL -> SKAB F1-Score: {metrics['f1']:.4f}")

            print("     [2/2] Training on SKAB, Testing on BATADAL...")
            cross_models_skab = {
                "LSTM": LSTMAnomalyDetector(1, config['deep_learning']['hidden_units'], config['deep_learning']['dropout_rate']),
                "1D-CNN": CNN1DAnomalyDetector(1, config['deep_learning']['hidden_units'], config['deep_learning']['dropout_rate'])
            }
            
            train_dataset_skab_cross = TimeSeriesDataset(skab_splits[0]['X_train'], skab_splits[0]['y_train'], dl_window_size)
            train_loader_skab_cross = DataLoader(train_dataset_skab_cross, batch_size=batch_size, shuffle=True)
            
            for model_name, model in cross_models_skab.items():
                trained_cross = train_model(model, train_loader_skab_cross, config)
                metrics = evaluate_model(trained_cross, test_loader)
                results_log["CROSS_DATASET"][seed]["Train_SKAB_Test_BATADAL"][model_name] = metrics
                print(f"        [{model_name}] SKAB -> BATADAL F1-Score: {metrics['f1']:.4f}")

    results_path = os.path.join(config["output_dir"], "dl_results.json")
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results_log, file, indent=4)

    print(f"\n✅ All runs complete! Results saved to {results_path}")

if __name__ == "__main__":
    main()