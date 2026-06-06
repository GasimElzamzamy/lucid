import json
import os

from src.data_pipeline import load_config, load_and_prepare_batadal
from src.models_automata import ProbabilisticAutomata


def main():
    config = load_config()
    os.makedirs(config["output_dir"], exist_ok=True)

    data = load_and_prepare_batadal(config)

    model = ProbabilisticAutomata(
        window_size=5,
        alphabet_size=3,
        laplace_smoothing=config["automata"]["laplace_smoothing"],
    )

    model.fit(data["X_train_pc1"])

    _ = model.predict(
        data["X_test_pc1"],
        anomaly_threshold=config["automata"]["anomaly_threshold"],
    )

    anomaly_explanations = [
        explanation
        for explanation in model.last_explanations
        if explanation["decision"] == "anomaly"
    ]

    if anomaly_explanations:
        sample = anomaly_explanations[0]
    else:
        sample = model.last_explanations[0]

    output = {
        "dataset": "BATADAL",
        "model": "ProbabilisticAutomata",
        "window_size": 5,
        "alphabet_size": 3,
        "explanation": sample,
    }

    output_path = os.path.join(
        config["output_dir"],
        "sample_explanation.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(f"Saved sample explanation to {output_path}")


if __name__ == "__main__":
    main()