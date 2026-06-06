import json


class AutomataExplainer:
    """Small JSON formatter for probabilistic automata decisions."""

    def __init__(self, anomaly_threshold):
        self.anomaly_threshold = float(anomaly_threshold)

    def explain_step(self, time_step, state, pattern, status, mapped_to,
                     probability, transitions=None):
        decision = "anomaly" if probability < self.anomaly_threshold else "normal"

        return {
            "time_step": int(time_step),
            "state": state,
            "pattern": pattern,
            "status": status,
            "mapped_to": mapped_to,
            "transitions": transitions or [],
            "probability": float(probability),
            "decision": decision,
            "confidence_score": float(probability),
        }

    def explain_path(self, time_step, sequence, probability, transitions):
        last_pattern = sequence[-1] if sequence else None
        last_transition = transitions[-1] if transitions else {}
        status = last_transition.get("to_status", "seen")
        mapped_to = last_transition.get("to_state") if status == "unseen" else None

        return self.explain_step(
            time_step=time_step,
            state=last_transition.get("from_state"),
            pattern=last_pattern,
            status=status,
            mapped_to=mapped_to,
            probability=probability,
            transitions=transitions,
        )

    @staticmethod
    def to_json(explanation, indent=2):
        return json.dumps(explanation, indent=indent, ensure_ascii=False)
