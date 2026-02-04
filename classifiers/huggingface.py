from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

from utils.data import load_data


class HuggingFaceModel:
    def __init__(self, model_name: str = "camembert-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.labels_: List[str] = []
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["description"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def train(self, csv_path: str, test_size: float = 0.2, num_epochs: int = 10,
              batch_size: int = 8, learning_rate: float = 2e-5) -> float | None:
        df = load_data(csv_path, require_disease=True)
        df = df[df["disease"].notna() & (df["disease"] != "")]
        if df.empty:
            raise ValueError("Aucune donnée utile pour entraîner")

        texts = df["description"].tolist()
        labels = df["disease"].tolist()
        self.labels_ = sorted(list(set(labels)))
        self.label2id = {l: i for i, l in enumerate(self.labels_)}
        self.id2label = {i: l for i, l in enumerate(self.labels_)}
        y_idx = [self.label2id[l] for l in labels]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels_),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)

        if test_size > 0:
            x_tr, x_te, y_tr, y_te = train_test_split(
                texts, y_idx, test_size=test_size, stratify=y_idx, random_state=42
            )
        else:
            x_tr, x_te, y_tr, y_te = texts, texts, y_idx, y_idx

        train_ds = Dataset.from_dict({"description": x_tr, "label": y_tr}).map(self._tokenize, batched=True)
        test_ds = Dataset.from_dict({"description": x_te, "label": y_te}).map(self._tokenize, batched=True)

        args = TrainingArguments(
            output_dir="./models/hf_training_output",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch" if test_size > 0 else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if test_size > 0 else False,
            metric_for_best_model="accuracy" if test_size > 0 else None,
            save_total_limit=2,
            seed=42,
        )

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            return {"accuracy": accuracy_score(labels, preds)}

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds if test_size > 0 else None,
            compute_metrics=compute_metrics if test_size > 0 else None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if test_size > 0 else None,
        )
        trainer.train()

        if test_size > 0 and len(x_te) > 0:
            results = trainer.evaluate()
            return results.get("eval_accuracy")
        return None

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(p)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(p)
        meta = {
            "labels": self.labels_,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "model_name": self.model_name,
        }
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        p = Path(path)
        with open(p / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.labels_ = meta["labels"]
        self.label2id = meta["label2id"]
        self.id2label = {int(k): v for k, v in meta["id2label"].items()}
        self.model_name = meta.get("model_name", "camembert-base")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(p)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            p, id2label=self.id2label, label2id=self.label2id
        )
        self.model.to(self.device)
        self.model.eval()

    def predict_list(self, texts: List[str]) -> pd.DataFrame:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modèle non chargé")

        rows = []
        batch = 16
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i + batch]
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1)
                conf = torch.max(probs, dim=-1)[0]

            for j, (idx, c) in enumerate(zip(pred_idx, conf)):
                rows.append({
                    "id": i + j + 1,
                    "prediction": self.id2label[idx.item()],
                    "confidence": round(c.item() * 100.0, 2),
                })
        return pd.DataFrame(rows)

