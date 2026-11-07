# Next Word Prediction — RNN, LSTM & GRU

A beginner-friendly Flask web app that trains three recurrent neural networks on the Tiny Shakespeare text dataset and compares their next-word predictions.

## 1. Project structure

```text
next_word_prediction_project/
├── app.py
├── train.py
├── download_dataset.py
├── requirements.txt
├── data/                 # dataset downloaded here
├── models/               # trained models and tokenizer created here
├── templates/index.html
└── static/style.css
```

## 2. Dataset

**Tiny Shakespeare**: about 1.1 MB of English text collected from Shakespeare's plays. It is useful for a small educational language-modeling experiment, but its old literary style means it will not behave like a modern general-purpose autocomplete system.

Dataset source:
- Hugging Face dataset card: https://huggingface.co/datasets/karpathy/tiny_shakespeare
- Raw text file: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Run `python download_dataset.py` to download `data/tinyshakespeare.txt`.

## 3. Working flow

1. **Collect text** — download the text corpus.
2. **Tokenization** — map words to integer IDs.
3. **Create sequences** — use previous words as X and the following word as y.
4. **Train/validation split** — first 90% of sequences for training and remaining 10% for validation.
5. **Train three models** — SimpleRNN, LSTM and GRU use the same vocabulary, sequence length, split and training settings.
6. **Evaluate** — save best validation loss, validation accuracy and epochs trained.
7. **Save artifacts** — models are saved as `.keras`; tokenizer and settings are saved as JSON.
8. **Predict** — Flask converts entered text into IDs, pads it, gets model probabilities and shows the top suggestions.

### Model concept

For a phrase such as `to be or not to`, the model estimates a probability distribution for the next word. Top-k words are sorted by their predicted probability. The scores are model estimates, not guaranteed real-world confidence.

## 4. Setup on Windows

Open PowerShell in this project folder. Python 3.10 or 3.11 is recommended for a smoother TensorFlow setup.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## 5. Train all models

```powershell
python train.py
```

Training time depends on CPU/GPU, TensorFlow build, and the number of epochs. Default is 8 epochs. For a quick smoke test, use:

```powershell
$env:EPOCHS=1
python train.py
```

The project creates:
- `models/rnn.keras`
- `models/lstm.keras`
- `models/gru.keras`
- `models/tokenizer.json`
- `models/config.json`

Do not expect the three models to have identical metrics. Compare the validation results produced on your own machine.

## 6. Run the web app

After training completes:

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Type a phrase, select RNN/LSTM/GRU, and click **Predict next word**.

## 7. Troubleshooting

- **Models are not trained yet**: run `python train.py` before starting the web app.
- **Dataset download fails**: check internet access and retry `python download_dataset.py`.
- **TensorFlow install error**: use a supported Python version (3.10/3.11 recommended) and create a fresh virtual environment.
- **Unknown words**: this model only knows the vocabulary learned from the dataset; try a phrase in Shakespearean English.
- **Weak predictions**: this is an educational small-corpus model. More relevant and larger licensed text improves domain relevance, but requires more training resources.
- **Port already in use**: stop the other Flask process or change the port at the bottom of `app.py`.

## 8. Important evaluation note

This starter project creates overlapping sequences from the corpus and splits them in sequence order. Because adjacent sequences overlap, validation examples can share words/context with training examples. Treat validation metrics as an educational comparison, not a rigorous benchmark. For a more robust experiment, split the original text into train/validation portions **before** generating sequences.
