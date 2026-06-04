from src.models_automata import ProbabilisticAutomata
from src.explainer import AutomataExplainer


def test_handle_unseen_pattern_maps_to_nearest_known_state():
    model = ProbabilisticAutomata(window_size=3, alphabet_size=3)
    model.vocabulary = {"abc", "abb", "ccc"}

    mapped = model._handle_unseen_pattern("abd")

    assert mapped == "abb"


def test_path_probability_uses_smoothing_for_missing_transition():
    model = ProbabilisticAutomata(window_size=3, alphabet_size=3, laplace_smoothing=1e-5)
    model.vocabulary = {"aaa", "aab", "abb"}
    model.transition_counts = {"aaa": {"aab": 2}}
    model.transition_matrix = model._counts_to_probabilities(model.transition_counts)

    probability, transitions = model.calculate_path_probability(["aaa", "abb"])

    assert probability > 0
    assert transitions[0]["from_state"] == "aaa"
    assert transitions[0]["to_state"] == "abb"


def test_explainer_outputs_required_json_fields():
    explainer = AutomataExplainer(anomaly_threshold=0.2)

    explanation = explainer.explain_step(
        time_step=5,
        state="aab",
        pattern="adc",
        status="unseen",
        mapped_to="abc",
        probability=0.108,
    )

    assert explanation["time_step"] == 5
    assert explanation["state"] == "aab"
    assert explanation["pattern"] == "adc"
    assert explanation["status"] == "unseen"
    assert explanation["mapped_to"] == "abc"
    assert explanation["decision"] == "anomaly"
    assert explanation["confidence_score"] == 0.108
