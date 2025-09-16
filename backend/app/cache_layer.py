# backend/app/cache_layer.py
from diskcache import Cache
import hashlib
import json

cache = Cache(directory="./cache_dir")

def make_cache_key(query, top_k, alpha):
    key_raw = json.dumps({"q": query, "k": int(top_k), "alpha": float(alpha)}, sort_keys=True)
    return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

def get_cached(key):
    return cache.get(key, default=None)

def set_cached(key, value, expire=3600):
    # value should be JSON-serializable
    cache.set(key, value, expire=expire)
