import argparse
from pathlib import Path

from classifiers.textblob import TextBlobModel
from classifiers.spacy import SpacyModel
from classifiers.huggingface import HuggingFaceModel


def get_model(name: str):
    name = name.lower()
    if name == "textblob":
        return TextBlobModel()
    if name == "spacy":
        return SpacyModel()
    if name in {"hf", "huggingface"}:
        return HuggingFaceModel()
    raise ValueError("Modèle inconnu: textblob | spacy | hf")


def main():
    parser = argparse.ArgumentParser(description="Entraîner un modèle NLP")
    parser.add_argument("--model", required=True, choices=["textblob", "spacy", "hf"])
    parser.add_argument("--input", required=True, help="CSV avec description + disease")
    parser.add_argument("--output", required=True, help="Chemin du modèle sauvegardé")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    model = get_model(args.model)
    acc = model.train(args.input, test_size=args.test_size)
    model.save(args.output)

    if acc is not None:
        print(f"Accuracy (validation): {acc:.2%}")


if __name__ == "__main__":
    main()

