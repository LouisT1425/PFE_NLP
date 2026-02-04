import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

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


def save_cm_png(cm, labels, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Comparaison NLP")
    parser.add_argument("--input", required=True, help="CSV de test (avec disease)")
    parser.add_argument("--models-dir", required=True, help="Dossier contenant les modèles")
    parser.add_argument("--output-dir", default="prediction")
    args = parser.parse_args()

    df = load_data(args.input, require_disease=True)
    texts = df["description"].tolist()
    y_true = df["disease"].tolist()
    labels = sorted(list(set(y_true)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in ["textblob", "spacy", "hf"]:
        model = get_model(name)
        model_path = Path(args.models_dir) / name
        if name == "textblob":
            model_path = model_path.with_suffix(".joblib")
        model.load(str(model_path))

        preds = model.predict_list(texts)["prediction"].tolist()

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, average="macro", zero_division=0)
        rec = recall_score(y_true, preds, average="macro", zero_division=0)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)

        rows.append({
            "model": name,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
        })

        cm_norm = confusion_matrix(y_true, preds, labels=labels, normalize="true") * 100
        save_cm_png(cm_norm, labels, out_dir / f"confusion_matrix_normalized_{name}.png")

    pd.DataFrame(rows).to_csv(out_dir / "comparison_summary.csv", index=False)
    print(f"Résultats sauvegardés dans {out_dir}")


if __name__ == "__main__":
    main()

