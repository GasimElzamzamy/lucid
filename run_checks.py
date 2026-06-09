import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable

COMMANDS = [
    ("Import Check",
     [PYTHON, "-c", "import numpy, pandas, scipy, sklearn, torch, tslearn; print('imports ok')"]),

    ("Compile Check",
     [PYTHON, "-m", "py_compile", "main.py", "src/models_dl.py", "src/models_automata.py"]),

    ("Unit Tests",
     [PYTHON, "-m", "pytest"]),

    ("Main Experiment",
     [PYTHON, "main.py"]),

    ("Results Summary",
     [PYTHON, "-m", "src.results_summary"]),

    ("Statistical Tests",
     [PYTHON, "-m", "src.statistical_tests"]),

    ("Cross Dataset Evaluation",
     [PYTHON, "-m", "src.cross_dataset_evaluation"]),

    ("Generate Explanation Sample",
     [PYTHON, "-m", "src.generate_explanation_sample"]),
]


EXPECTED_FILES = [
    "results/dl_results.json",
    "results/summary_metrics.md",
    "results/best_automata_params.csv",
    "results/wilcoxon_results.md",
    "results/cross_dataset_results.md",
    "results/sample_explanation.json",
    "results/automata_state_graph.png",
    "results/automata_transition_heatmap.png",
]


def run_step(name, command):
    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print("=" * 80)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nFAILED: {name}")
        sys.exit(result.returncode)

    print(f"\nPASSED: {name}")


def check_files():
    print("\n" + "=" * 80)
    print("CHECKING GENERATED FILES")
    print("=" * 80)

    missing = []

    for file in EXPECTED_FILES:
        if Path(file).exists():
            print(f"[OK] {file}")
        else:
            print(f"[MISSING] {file}")
            missing.append(file)

    if missing:
        print("\nMissing files:")
        for m in missing:
            print(" -", m)
        sys.exit(1)

    print("\nAll expected files exist.")


def main():
    print("=" * 80)
    print("LUCID FINAL VALIDATION PIPELINE")
    print("=" * 80)
    print(f"Python executable: {PYTHON}")

    for name, cmd in COMMANDS:
        run_step(name, cmd)

    check_files()

    print("\n" + "=" * 80)
    print("ALL CHECKS PASSED")
    print("PROJECT READY FOR REPORTING")
    print("=" * 80)


if __name__ == "__main__":
    main()