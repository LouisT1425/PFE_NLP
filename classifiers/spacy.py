from pathlib import Path
from typing import List
import random

import pandas as pd
import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.data import load_data


class SpacyModel:
    def __init__(self):
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            self.nlp = spacy.blank("fr")

        if "lemmatizer" in self.nlp.pipe_names:
            self.nlp.remove_pipe("lemmatizer")
        if "textcat" in self.nlp.pipe_names:
            self.nlp.remove_pipe("textcat")

        self.textcat = self.nlp.add_pipe("textcat")
        if "exclusive_classes" in self.textcat.cfg:
            self.textcat.cfg["exclusive_classes"] = True
        if "multi_label" in self.textcat.cfg:
            self.textcat.cfg["multi_label"] = False

        self.labels_: List[str] = []

    def _make_examples(self, texts: List[str], y: List[int]) -> List[Example]:
        examples = []
        for text, idx in zip(texts, y):
            doc = self.nlp.make_doc(text)
            label = self.labels_[idx]
            cats = {lab: (lab == label) for lab in self.labels_}
            examples.append(Example.from_dict(doc, {"cats": cats}))
        return examples

    def train(self, csv_path: str, test_size: float = 0.2, n_iter: int = 10, batch_size: int = 8) -> float | None:
        df = load_data(csv_path, require_disease=True)
        df = df[df["disease"].notna() & (df["disease"] != "")]
        if df.empty:
            raise ValueError("Aucune donnée utile pour entraîner")

        texts = df["description"].tolist()
        labels = df["disease"].tolist()
        self.labels_ = sorted(list(set(labels)))
        for lab in self.labels_:
            self.textcat.add_label(lab)

        y_idx = [self.labels_.index(d) for d in labels]

        if test_size > 0:
            x_tr, x_te, y_tr, y_te = train_test_split(
                texts, y_idx, test_size=test_size, stratify=y_idx, random_state=42
            )
        else:
            x_tr, x_te, y_tr, y_te = texts, texts, y_idx, y_idx

        train_examples = self._make_examples(x_tr, y_tr)
        optimizer = self.nlp.initialize(lambda: train_examples)

        for _ in range(n_iter):
            random.shuffle(train_examples)
            for batch in minibatch(train_examples, size=batch_size):
                self.nlp.update(batch, sgd=optimizer)

        if test_size > 0 and len(x_te) > 0:
            preds = []
            for doc in self.nlp.pipe(x_te, disable=[p for p in self.nlp.pipe_names if p != "textcat"]):
                pred = max(doc.cats, key=doc.cats.get) if doc.cats else self.labels_[0]
                preds.append(self.labels_.index(pred))
            return accuracy_score(y_te, preds)
        return None

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(p)

    def load(self, path: str) -> None:
        p = Path(path)
        self.nlp = spacy.load(p)
        self.textcat = self.nlp.get_pipe("textcat")
        self.labels_ = list(self.textcat.labels)

    def predict_list(self, texts: List[str]) -> pd.DataFrame:
        rows = []
        for i, doc in enumerate(self.nlp.pipe(texts, disable=[p for p in self.nlp.pipe_names if p != "textcat"])):
            if doc.cats:
                label = max(doc.cats, key=doc.cats.get)
                conf = float(doc.cats[label]) * 100.0
            else:
                label = "N/A"
                conf = 0.0
            rows.append({"id": i + 1, "prediction": label, "confidence": round(conf, 2)})
        return pd.DataFrame(rows)

