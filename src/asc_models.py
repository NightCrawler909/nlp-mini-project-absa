import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from datasets import Dataset
import logging
from torch.utils.data import DataLoader, TensorDataset
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
inv_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

# ----------------- Baselines ----------------- #

def extract_aspect_window(text, aspect, window_size=5):
    """Extracts a window of words around the aspect."""
    words = text.split()
    aspect_words = aspect.split()
    try:
        # Simple heuristic: find first occurrence
        start_idx = words.index(aspect_words[0])
        end_idx = start_idx + len(aspect_words)
        left = max(0, start_idx - window_size)
        right = min(len(words), end_idx + window_size)
        return " ".join(words[left:right])
    except ValueError:
        return text # fallback if exact token match fails

def train_baseline_asc(train_df, val_df):
    train_df = train_df[train_df['polarity'].isin(label_map.keys())]
    val_df = val_df[val_df['polarity'].isin(label_map.keys())]
    
    # Feature: TF-IDF of sentence + aspect window
    train_texts = [row['text'] + " " + extract_aspect_window(row['text'], row['aspect']) for _, row in train_df.iterrows()]
    val_texts = [row['text'] + " " + extract_aspect_window(row['text'], row['aspect']) for _, row in val_df.iterrows()]
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    
    y_train = [label_map[p] for p in train_df['polarity']]
    y_val = [label_map[p] for p in val_df['polarity']]
    
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    
    svm = LinearSVC(class_weight='balanced')
    svm.fit(X_train, y_train)
    
    return vectorizer, lr, svm

# ----------------- Transformers ----------------- #

def train_transformer_asc(model_name, train_df, val_df, output_dir, use_class_weights=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Sentence-pair formulation: sentence [SEP] aspect
    train_df = train_df[train_df['polarity'].isin(label_map.keys())]
    val_df = val_df[val_df['polarity'].isin(label_map.keys())]
    
    def tokenize_data(df):
        texts = df['text'].tolist()
        aspects = df['aspect'].tolist()
        labels = [label_map[p] for p in df['polarity'].tolist()]
        
        encodings = tokenizer(texts, aspects, truncation=True, padding=True, max_length=128)
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        return dataset

    train_dataset = tokenize_data(train_df)
    val_dataset = tokenize_data(val_df)
    
    # Class weights calculation
    if use_class_weights:
        class_counts = train_df['polarity'].value_counts()
        total = len(train_df)
        weights = [total / class_counts.get(inv_label_map[i], 1) for i in range(3)]
        weights = torch.tensor(weights, dtype=torch.float32)
    else:
        weights = None

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=weights.to(model.device))
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        return {"accuracy": acc, "f1": f1}

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
        save_strategy="no",
        load_best_model_at_end=False
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    logging.info(f"Training time for {model_name}: {train_time} seconds")
    
    return trainer, model, tokenizer

# ----------------- Reference Model ----------------- #

def evaluate_reference_model(val_df):
    """Evaluates the external pre-trained baseline yangheng/deberta-v3-base-absa-v1.1"""
    # This pipeline expects inputs like: "I love this [ASP] laptop [ASP]" or just aspect based sentiment
    # We will format it as [CLS] text [SEP] aspect [SEP] or use the standard pipeline
    try:
        classifier = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
        val_df = val_df[val_df['polarity'].isin(label_map.keys())]
        
        inputs = [f"[CLS] {row['text']} [SEP] {row['aspect']} [SEP]" for _, row in val_df.iterrows()]
        
        preds = []
        for out in classifier(inputs):
            # Model outputs Negative, Neutral, Positive
            lbl = out['label'].lower()
            preds.append(label_map.get(lbl, 1))
            
        labels = [label_map[p] for p in val_df['polarity']]
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return {"accuracy": acc, "f1": f1}
    except Exception as e:
        logging.error(f"Failed to load/eval reference model: {e}")
        return None

if __name__ == "__main__":
    pass
