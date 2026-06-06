import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data_pipeline import load_config, load_and_prepare_batadal
from src.models_automata import ProbabilisticAutomata


def build_transition_matrix_array(automata, max_states=20):
    """
    Converts automata.transition_matrix dictionary into a dense matrix
    for visualization. Only the most frequent states are shown to keep
    the heatmap readable.
    """
    state_frequency = {
        state: sum(next_states.values())
        for state, next_states in automata.transition_counts.items()
    }

    selected_states = sorted(
        state_frequency,
        key=state_frequency.get,
        reverse=True
    )[:max_states]

    matrix = np.zeros((len(selected_states), len(selected_states)))

    for i, from_state in enumerate(selected_states):
        for j, to_state in enumerate(selected_states):
            matrix[i, j] = automata._transition_probability(from_state, to_state)

    return selected_states, matrix


def plot_transition_heatmap(automata, output_path):
    states, matrix = build_transition_matrix_array(automata)

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        matrix,
        xticklabels=states,
        yticklabels=states,
        cmap="viridis",
        linewidths=0.3
    )

    plt.title("Probabilistic Automata Transition Probability Heatmap")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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

    output_path = os.path.join(
        config["output_dir"],
        "automata_transition_heatmap.png"
    )

    plot_transition_heatmap(model, output_path)

    print(f"Saved transition heatmap to {output_path}")


if __name__ == "__main__":
    main()