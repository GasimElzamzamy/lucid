import os

import matplotlib.pyplot as plt
import networkx as nx

from src.data_pipeline import load_config, load_and_prepare_batadal
from src.models_automata import ProbabilisticAutomata


def build_graph(automata, max_edges=50):
    G = nx.DiGraph()

    edges = []

    for from_state, next_states in automata.transition_counts.items():
        for to_state, count in next_states.items():
            edges.append((from_state, to_state, count))

    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    edges = edges[:max_edges]

    for from_state, to_state, count in edges:
        G.add_edge(from_state, to_state, weight=count)

    return G


def main():
    config = load_config()

    data = load_and_prepare_batadal(config)

    automata = ProbabilisticAutomata(
        window_size=5,
        alphabet_size=3,
        laplace_smoothing=config["automata"]["laplace_smoothing"]
    )

    automata.fit(data["X_train_pc1"])

    G = build_graph(automata)

    plt.figure(figsize=(16, 12))

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.0
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1200
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True
    )

    plt.title("Probabilistic Automata State Transition Graph")

    output_path = os.path.join(
        config["output_dir"],
        "automata_state_graph.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved graph to {output_path}")


if __name__ == "__main__":
    main()