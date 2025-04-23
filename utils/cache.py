import os
import json
import hashlib
import pickle
from datetime import datetime
import shutil

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache')


def ensure_cache_dir():
    """Ensure cache directory exists"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(key):
    """Get path for cache file"""
    # Create a hash of the key to use as filename
    hash_obj = hashlib.md5(key.encode())
    filename = hash_obj.hexdigest() + '.pkl'
    return os.path.join(CACHE_DIR, filename)


def check_cache(key):
    """Check if data exists in cache"""
    ensure_cache_dir()
    cache_path = get_cache_path(key)

    if not os.path.exists(cache_path):
        return False

    # Load metadata to check if cache is valid
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        # Check metadata for validity
        metadata = cache_data.get('metadata', {})

        # Check if cache has expired
        if 'expiry' in metadata:
            expiry = datetime.fromisoformat(metadata['expiry'])
            if datetime.now() > expiry:
                return False

        return True
    except:
        return False


def cache_data(key, data, expiry_days=None):
    """Cache data with optional expiry"""
    ensure_cache_dir()
    cache_path = get_cache_path(key)

    # Add metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'key': key
    }

    if expiry_days:
        expiry = datetime.now() + timedelta(days=expiry_days)
        metadata['expiry'] = expiry.isoformat()

    # Prepare data for caching
    cache_data = {
        'data': data,
        'metadata': metadata
    }

    # Save to cache
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)


def get_cached_data(key):
    """Get data from cache"""
    if not check_cache(key):
        return None

    cache_path = get_cache_path(key)

    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data['data']
    except:
        return None


def clear_cache(key=None):
    """Clear cache for specific key or all cache"""
    ensure_cache_dir()

    if key:
        cache_path = get_cache_path(key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
    else:
        # Clear all cache files
        for file in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, file)
            if os.path.isfile(file_path) and file.endswith('.pkl'):
                os.remove(file_path)