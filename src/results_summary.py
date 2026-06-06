import json
import os
import pandas as pd


RESULTS_PATH = os.path.join("results", "dl_results.json")
SUMMARY_CSV_PATH = os.path.join("results", "summary_metrics.csv")
BEST_AUTOMATA_CSV_PATH = os.path.join("results", "best_automata_params.csv")
SUMMARY_MD_PATH = os.path.join("results", "summary_metrics.md")


def load_results(path=RESULTS_PATH):
    with open(path, "r") as file:
        return json.load(file)


def flatten_results(results):
    rows = []

    for dataset_name, dataset_results in results.items():
        for seed, seed_results in dataset_results.items():
            for model_name, model_results in seed_results.items():

                if model_name == "Automata":
                    for experiment_name, experiment_results in model_results.items():
                        for condition, metrics in experiment_results.items():
                            rows.append({
                                "dataset": dataset_name,
                                "seed": seed,
                                "model": model_name,
                                "experiment": experiment_name,
                                "condition": condition,
                                "f1": metrics.get("f1"),
                                "precision": metrics.get("precision"),
                                "recall": metrics.get("recall"),
                                "accuracy": metrics.get("accuracy"),
                                "f1_std": metrics.get("f1_std"),
                                "num_states": metrics.get("num_states"),
                                "num_transitions": metrics.get("num_transitions"),
                            })
                else:
                    for condition, metrics in model_results.items():
                        rows.append({
                            "dataset": dataset_name,
                            "seed": seed,
                            "model": model_name,
                            "experiment": "default",
                            "condition": condition,
                            "f1": metrics.get("f1"),
                            "precision": metrics.get("precision"),
                            "recall": metrics.get("recall"),
                            "accuracy": metrics.get("accuracy"),
                            "f1_std": metrics.get("f1_std"),
                            "num_states": None,
                            "num_transitions": None,
                        })

    return pd.DataFrame(rows)


def build_best_automata_table(summary_df):
    automata_df = summary_df[summary_df["model"] == "Automata"].copy()

    grouped = (
        automata_df
        .groupby(["dataset", "experiment", "condition"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_accuracy=("accuracy", "mean"),
        )
    )

    best_rows = []

    for (dataset, condition), group in grouped.groupby(["dataset", "condition"]):
        best = group.sort_values("mean_f1", ascending=False).iloc[0]
        best_rows.append(best)

    return pd.DataFrame(best_rows)


def build_markdown_summary(summary_df, best_automata_df):
    model_summary = (
        summary_df
        .groupby(["dataset", "model", "experiment", "condition"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
        )
        .sort_values(["dataset", "condition", "mean_f1"], ascending=[True, True, False])
    )

    lines = []
    lines.append("# LUCID Experiment Summary\n")
    lines.append("## Overall Model Performance\n")
    lines.append(model_summary.to_markdown(index=False))
    lines.append("\n\n## Best Automata Hyperparameters\n")
    lines.append(best_automata_df.to_markdown(index=False))

    return "\n".join(lines)


def main():
    os.makedirs("results", exist_ok=True)

    results = load_results()
    summary_df = flatten_results(results)

    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    best_automata_df = build_best_automata_table(summary_df)
    best_automata_df.to_csv(BEST_AUTOMATA_CSV_PATH, index=False)

    markdown_summary = build_markdown_summary(summary_df, best_automata_df)

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as file:
        file.write(markdown_summary)

    print(f"Saved summary table to {SUMMARY_CSV_PATH}")
    print(f"Saved best automata table to {BEST_AUTOMATA_CSV_PATH}")
    print(f"Saved markdown summary to {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()