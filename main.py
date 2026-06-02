import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

# Import our custom modules
from src.data_pipeline import load_config, load_skab, prepare_skab_cv, load_and_prepare_batadal
from src.models_dl import TimeSeriesDataset, LSTMAnomalyDetector, CNN1DAnomalyDetector

# ==========================================
# 1. UTILITIES & REPRODUCIBILITY
# ==========================================
def set_seed(seed):
    """Locks all random number generators for strict reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def apply_gaussian_noise(X, config):
    """Injects noise to test model robustness if enabled in config."""
    noise_cfg = config['scenarios']['gaussian_noise']
    if noise_cfg['apply']:
        noise = np.random.normal(noise_cfg['mean'], noise_cfg['std_dev'], X.shape)
        return X + noise
    return X

# ==========================================
# 2. TRAINING & EVALUATION ENGINES
# ==========================================
def train_model(model, train_loader, config):
    """Standard PyTorch training loop."""
    criterion = nn.BCELoss() # Binary Cross Entropy for Anomaly (0) vs Normal (1)
    optimizer = optim.Adam(model.parameters(), lr=config['deep_learning']['learning_rate'])
    epochs = config['deep_learning']['epochs']
    
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, test_loader):
    """Evaluates the model and returns F1, Precision, and Recall."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            # Threshold probabilities at 0.5 to get binary labels
            binary_preds = (predictions > 0.5).float()
            all_preds.extend(binary_preds.numpy())
            all_targets.extend(y_batch.numpy())
            
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    
    return {"f1": f1, "precision": precision, "recall": recall}

# ==========================================
# 3. MASTER ORCHESTRATION LOOP
# ==========================================
def main():
    config = load_config()
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # We will test on window_size=5 as a baseline for DL models
    window_size = 5 
    batch_size = config['deep_learning']['batch_size']
    
    # Load Data Once
    print("Loading datasets...")
    batadal_data = load_and_prepare_batadal(config)
    
    results_log = {"BATADAL": {}, "SKAB": {}}
    
    # The Rubric Loop: Run everything across the 5 specific seeds
    for seed in config['random_seeds']:
        print(f"\n--- Running Experiment for Seed: {seed} ---")
        set_seed(seed)
        
        # --- BATADAL EXPERIMENT ---
        print("Training on BATADAL...")
        train_dataset = TimeSeriesDataset(batadal_data['X_train'], batadal_data['y_train'], window_size)
        test_dataset = TimeSeriesDataset(batadal_data['X_test'], batadal_data['y_test'], window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize Models (43 features for BATADAL)
        input_size = batadal_data['X_train'].shape[1]
        models = {
            "LSTM": LSTMAnomalyDetector(input_size, config['deep_learning']['hidden_units'], config['deep_learning']['dropout_rate']),
            "1D-CNN": CNN1DAnomalyDetector(input_size, config['deep_learning']['hidden_units'], config['deep_learning']['dropout_rate'])
        }
        
        results_log["BATADAL"][seed] = {}
        for model_name, model in models.items():
            print(f"  -> Training {model_name}...")
            trained_model = train_model(model, train_loader, config)
            metrics = evaluate_model(trained_model, test_loader)
            results_log["BATADAL"][seed][model_name] = metrics
            print(f"     {model_name} F1-Score: {metrics['f1']:.4f}")

        # --- SKAB EXPERIMENT ---
        print("Training on SKAB (GroupKFold)...")
        skab_df = load_skab(config)
        skab_splits = prepare_skab_cv(skab_df, config)
        
        # Dictionary to hold the scores across the 5 folds
        fold_metrics = {"LSTM": {"f1": [], "precision": [], "recall": []},
                        "1D-CNN": {"f1": [], "precision": [], "recall": []}}
                        
        for fold_idx, fold_data in enumerate(skab_splits):
            print(f"  -> SKAB Fold {fold_idx + 1}/{len(skab_splits)}")
            train_dataset_skab = TimeSeriesDataset(fold_data['X_train'], fold_data['y_train'], window_size)
            test_dataset_skab = TimeSeriesDataset(fold_data['X_test'], fold_data['y_test'], window_size)
            
            train_loader_skab = DataLoader(train_dataset_skab, batch_size=batch_size, shuffle=True)
            test_loader_skab = DataLoader(test_dataset_skab, batch_size=batch_size, shuffle=False)
            
            input_size_skab = fold_data['X_train'].shape[1] # SKAB has 8 features
            
            # Re-initialize models specifically for SKAB's input size
            models_skab = {
                "LSTM": LSTMAnomalyDetector(input_size_skab, config['deep_learning']['hidden_units'], config['deep_learning']['dropout_rate']),
                "1D-CNN": CNN1DAnomalyDetector(input_size_skab, config['deep_learning']['hidden_units'], config['deep_learning']['dropout_rate'])
            }
            
            for model_name, model in models_skab.items():
                trained_model = train_model(model, train_loader_skab, config)
                metrics = evaluate_model(trained_model, test_loader_skab)
                fold_metrics[model_name]["f1"].append(metrics["f1"])
                fold_metrics[model_name]["precision"].append(metrics["precision"])
                fold_metrics[model_name]["recall"].append(metrics["recall"])
                
        # Average the metrics across all folds for this specific seed
        results_log["SKAB"][seed] = {}
        for model_name in fold_metrics:
            avg_f1 = np.mean(fold_metrics[model_name]["f1"])
            avg_prec = np.mean(fold_metrics[model_name]["precision"])
            avg_rec = np.mean(fold_metrics[model_name]["recall"])
            results_log["SKAB"][seed][model_name] = {"f1": avg_f1, "precision": avg_prec, "recall": avg_rec}
            print(f"     SKAB {model_name} Avg F1-Score: {avg_f1:.4f}")

    # Save final results to JSON
    results_path = os.path.join(config['output_dir'], "dl_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_log, f, indent=4)
    print(f"\n✅ All runs complete! Results saved to {results_path}")

if __name__ == "__main__":
    main()