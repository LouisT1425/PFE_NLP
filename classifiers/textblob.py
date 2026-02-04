from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob.classifiers import NaiveBayesClassifier

from utils.data import load_data


class TextBlobModel:
    def __init__(self):
        self.clf: NaiveBayesClassifier | None = None
        self.labels_: List[str] = []

    @staticmethod
    def _make_training_data(texts: List[str], labels: List[str]) -> List[Tuple[str, str]]:
        return list(zip(texts, labels))

    def train(self, csv_path: str, test_size: float = 0.2) -> float | None:
        df = load_data(csv_path, require_disease=True)
        df = df[df["disease"].notna() & (df["disease"] != "")]
        if df.empty:
            raise ValueError("Aucune donnée utile pour entraîner")

        texts = df["description"].tolist()
        labels = df["disease"].tolist()
        self.labels_ = sorted(list(set(labels)))

        if test_size > 0:
            x_tr, x_te, y_tr, y_te = train_test_split(
                texts, labels, test_size=test_size, stratify=labels, random_state=42
            )
        else:
            x_tr, x_te, y_tr, y_te = texts, texts, labels, labels

        self.clf = NaiveBayesClassifier(self._make_training_data(x_tr, y_tr))

        if test_size > 0 and len(x_te) > 0:
            preds = [self.clf.classify(t) for t in x_te]
            return accuracy_score(y_te, preds)
        return None

    def save(self, path: str) -> None:
        if self.clf is None:
            raise RuntimeError("Modèle non entraîné")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "labels": self.labels_}, p)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.clf = data["clf"]
        self.labels_ = data.get("labels", [])

    def predict_list(self, texts: List[str]) -> pd.DataFrame:
        if self.clf is None:
            raise RuntimeError("Modèle non chargé")
        rows = []
        for i, text in enumerate(texts):
            label = self.clf.classify(text)
            prob = self.clf.prob_classify(text).prob(label) * 100.0
            rows.append({"id": i + 1, "prediction": label, "confidence": round(prob, 2)})
        return pd.DataFrame(rows)

