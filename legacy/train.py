import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from download_dataset import DEST, URL
from urllib.request import urlretrieve

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

VOCAB_LIMIT = 12000
MAX_LEN = 12
EPOCHS = int(os.getenv("EPOCHS", "8"))
BATCH_SIZE = 128
SEED = 42

tf.keras.utils.set_random_seed(SEED)

if not DEST.exists():
    print("Dataset not found; downloading it...")
    urlretrieve(URL, DEST)

text = DEST.read_text(encoding="utf-8").lower()
tokenizer = Tokenizer(num_words=VOCAB_LIMIT, oov_token="<unk>")
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = min(VOCAB_LIMIT, len(tokenizer.word_index) + 1)

# Each training example uses up to MAX_LEN preceding words to predict the next word.
examples = []
for i in range(2, len(sequences)):
    start = max(0, i - MAX_LEN)
    examples.append(sequences[start:i + 1])

examples = pad_sequences(examples, maxlen=MAX_LEN + 1, padding="pre")
X, y = examples[:, :-1], examples[:, -1]
split = int(len(X) * 0.9)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Text characters: {len(text):,}")
print(f"Vocabulary size: {vocab_size:,}")
print(f"Training examples: {len(X_train):,}; validation examples: {len(X_val):,}")

def build_model(kind):
    recurrent = {
        "RNN": tf.keras.layers.SimpleRNN,
        "LSTM": tf.keras.layers.LSTM,
        "GRU": tf.keras.layers.GRU,
    }[kind]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_LEN,)),
        tf.keras.layers.Embedding(vocab_size, 96, mask_zero=True),
        recurrent(128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

results = {}
for kind in ("RNN", "LSTM", "GRU"):
    print(f"\nTraining {kind}...")
    model = build_model(kind)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(MODEL_DIR / f"{kind.lower()}.keras")
    results[kind] = {
        "val_loss": float(min(history.history["val_loss"])),
        "val_accuracy": float(max(history.history["val_sparse_categorical_accuracy"])),
        "epochs_trained": len(history.history["loss"]),
    }

with open(MODEL_DIR / "tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())
with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump({"max_len": MAX_LEN, "vocab_size": vocab_size, "results": results}, f, indent=2)

print("\nTraining finished. Results:")
for name, metrics in results.items():
    print(f"{name}: val_accuracy={metrics['val_accuracy']:.4f}, val_loss={metrics['val_loss']:.4f}")
print("Start the website with: python app.py")
