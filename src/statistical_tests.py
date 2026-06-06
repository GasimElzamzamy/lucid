import json
import os

import pandas as pd
from scipy.stats import wilcoxon


RESULTS_PATH = os.path.join("results", "dl_results.json")
OUTPUT_CSV = os.path.join("results", "wilcoxon_results.csv")
OUTPUT_MD = os.path.join("results", "wilcoxon_results.md")


def load_results():
    with open(RESULTS_PATH, "r") as file:
        return json.load(file)


def extract_f1_scores(results, dataset, model, condition="clean", experiment=None):
    scores = []

    for seed in results[dataset]:
        model_data = results[dataset][seed][model]

        if model == "Automata":
            if experiment is None:
                raise ValueError("Automata requires an experiment name such as w5_a3.")
            scores.append(model_data[experiment][condition]["f1"])
        else:
            scores.append(model_data[condition]["f1"])

    return scores


def get_best_automata_experiment(results, dataset, condition="clean"):
    first_seed = next(iter(results[dataset]))
    experiments = results[dataset][first_seed]["Automata"].keys()

    best_experiment = None
    best_mean_f1 = -1

    for experiment in experiments:
        scores = extract_f1_scores(
            results=results,
            dataset=dataset,
            model="Automata",
            condition=condition,
            experiment=experiment,
        )
        mean_f1 = sum(scores) / len(scores)

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_experiment = experiment

    return best_experiment, best_mean_f1


def safe_wilcoxon(scores_a, scores_b):
    try:
        statistic, p_value = wilcoxon(scores_a, scores_b)
        return float(statistic), float(p_value)
    except ValueError:
        return None, None


def main():
    results = load_results()
    rows = []

    for dataset in ["BATADAL", "SKAB"]:
        best_automata, best_f1 = get_best_automata_experiment(
            results,
            dataset,
            condition="clean",
        )

        automata_scores = extract_f1_scores(
            results,
            dataset,
            model="Automata",
            condition="clean",
            experiment=best_automata,
        )

        for baseline_model in ["LSTM", "1D-CNN"]:
            baseline_scores = extract_f1_scores(
                results,
                dataset,
                model=baseline_model,
                condition="clean",
            )

            statistic, p_value = safe_wilcoxon(
                baseline_scores,
                automata_scores,
            )

            rows.append({
                "dataset": dataset,
                "comparison": f"{baseline_model} vs Automata",
                "automata_experiment": best_automata,
                "baseline_mean_f1": sum(baseline_scores) / len(baseline_scores),
                "automata_mean_f1": sum(automata_scores) / len(automata_scores),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "significant_at_0.05": (
                    None if p_value is None else p_value < 0.05
                ),
            })

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_MD, "w", encoding="utf-8") as file:
        file.write("# Wilcoxon Signed-Rank Test Results\n\n")
        file.write(df.to_markdown(index=False))

    print(f"Saved Wilcoxon CSV to {OUTPUT_CSV}")
    print(f"Saved Wilcoxon Markdown to {OUTPUT_MD}")


if __name__ == "__main__":
    main()