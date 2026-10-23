import os
import sys

# repomind reads these at import time; tests never call an API, and an API-style embedding model
# makes token counting use tiktoken instead of downloading the local embedder's tokenizer
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["EMBED_MODEL"] = "text-embedding-3-small"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "eval")]
