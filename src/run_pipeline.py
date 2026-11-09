"""Relance le pipeline de bout en bout, des données brutes aux résultats finaux."""

import argparse
import logging

from src import build_dataset, eda, evaluate, robustness, train


def main():
    """Exécute les étapes dans l'ordre ; la robustesse est facultative car longue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-robustness",
        action="store_true",
        help="ajoute la validation imbriquée et la validation par centre (environ 30 minutes)",
    )
    arguments = parser.parse_args()

    steps = [build_dataset, eda, train, evaluate]
    if arguments.with_robustness:
        steps.append(robustness)
    for step in steps:
        step.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
