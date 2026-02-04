import argparse
import pandas as pd

from classifiers.textblob import TextBlobModel
from classifiers.spacy import SpacyModel
from classifiers.huggingface import HuggingFaceModel
from utils.data import load_data


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
    parser = argparse.ArgumentParser(description="Prédire avec un modèle NLP")
    parser.add_argument("--model", required=True, choices=["textblob", "spacy", "hf"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True, help="CSV avec description")
    parser.add_argument("--output", required=True, help="CSV de sortie")
    args = parser.parse_args()

    model = get_model(args.model)
    model.load(args.model_path)

    df = load_data(args.input, require_disease=False)
    preds = model.predict_list(df["description"].tolist())
    out = pd.concat([df.reset_index(drop=True), preds], axis=1)
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Prédictions sauvegardées: {args.output}")


if __name__ == "__main__":
    main()

