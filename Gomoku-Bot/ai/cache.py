from cachetools import LRUCache
from config import config

class Cache:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.cache = LRUCache(maxsize=capacity)
        self.enable_cache = config['enableCache']

    def get(self, key):
        if not self.enable_cache:
            return None
        return self.cache.get(key, None)

    def put(self, key, value):
        if not self.enable_cache:
            return
        self.cache[key] = value

    def has(self, key):
        if not self.enable_cache:
            return False
        return key in self.cache