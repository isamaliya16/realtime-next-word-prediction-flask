from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = DATA_DIR / "tinyshakespeare.txt"

if not DEST.exists():
    print("Downloading Tiny Shakespeare text dataset...")
    urlretrieve(URL, DEST)
print(f"Dataset ready: {DEST} ({DEST.stat().st_size:,} bytes)")
