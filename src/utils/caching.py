"""
Advanced caching utilities for liquid neural networks.
"""

import pickle
import hashlib
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
import warnings

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    import numpy as jnp


class AdaptiveCache:
    """
    Adaptive caching system that learns from access patterns.
    """
    
    def __init__(self, max_size: int = 1000, cache_dir: Optional[str] = None):
        self.max_size = max_size
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.memory_cache = {}
        self.access_counts = {}
        self.access_times = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': str(args),
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _should_cache_to_disk(self, key: str) -> bool:
        """Decide if item should be cached to disk based on access patterns."""
        access_count = self.access_counts.get(key, 0)
        return access_count > 3  # Cache to disk if accessed more than 3 times
    
    def _evict_least_used(self):
        """Evict least recently used items."""
        if len(self.memory_cache) < self.max_size:
            return
            
        # Sort by access time
        sorted_items = sorted(
            self.access_times.items(), 
            key=lambda x: x[1]
        )
        
        # Remove oldest 25% of items
        items_to_remove = len(sorted_items) // 4
        for key, _ in sorted_items[:items_to_remove]:
            if key in self.memory_cache:
                # Save to disk if frequently accessed
                if self._should_cache_to_disk(key):
                    self._save_to_disk(key, self.memory_cache[key])
                    
                del self.memory_cache[key]
                del self.access_times[key]
                self.cache_stats['evictions'] += 1
    
    def _save_to_disk(self, key: str, value: Any):
        """Save item to disk."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            warnings.warn(f"Failed to save cache item to disk: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load item from disk."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load cache item from disk: {e}")
        return None
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        # Check memory cache first
        if key in self.memory_cache:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.access_times[key] = time.time()
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]
        
        # Check disk cache
        disk_value = self._load_from_disk(key)
        if disk_value is not None:
            # Move back to memory cache
            self.memory_cache[key] = disk_value
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.access_times[key] = time.time()
            self.cache_stats['hits'] += 1
            return disk_value
        
        self.cache_stats['misses'] += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        self._evict_least_used()
        
        self.memory_cache[key] = value
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        self.access_times[key] = time.time()
    
    def invalidate(self, key: str):
        """Invalidate cache entry."""
        if key in self.memory_cache:
            del self.memory_cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.access_counts:
            del self.access_counts[key]
            
        # Remove from disk
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
    
    def clear(self):
        """Clear all cache."""
        self.memory_cache.clear()
        self.access_counts.clear()
        self.access_times.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'memory_items': len(self.memory_cache),
            'disk_items': len(list(self.cache_dir.glob("*.pkl"))),
            'total_requests': total_requests
        }


class ComputationCache:
    """Cache for expensive computations in liquid neural networks."""
    
    def __init__(self, cache_dir: str = "computation_cache"):
        self.cache = AdaptiveCache(max_size=500, cache_dir=cache_dir)
        
    def cached_forward_pass(self, forward_fn: Callable) -> Callable:
        """Cache forward pass results."""
        def wrapper(inputs, hidden_state, dt=0.1):
            # Create cache key from inputs
            try:
                if HAS_JAX:
                    input_hash = hashlib.md5(inputs.tobytes()).hexdigest()[:8]
                    hidden_hash = hashlib.md5(hidden_state.tobytes()).hexdigest()[:8]
                else:
                    input_hash = hashlib.md5(str(inputs).encode()).hexdigest()[:8]
                    hidden_hash = hashlib.md5(str(hidden_state).encode()).hexdigest()[:8]
                    
                cache_key = f"forward_{input_hash}_{hidden_hash}_{dt}"
                
                # Check cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Compute and cache
                result = forward_fn(inputs, hidden_state, dt)
                self.cache.put(cache_key, result)
                return result
                
            except Exception:
                # If caching fails, just compute normally
                return forward_fn(inputs, hidden_state, dt)
                
        return wrapper
    
    def cached_stability_analysis(self, stability_fn: Callable) -> Callable:
        """Cache stability analysis results."""
        def wrapper(model):
            try:
                # Create cache key from model parameters
                if hasattr(model, 'W_rec'):
                    param_hash = hashlib.md5(model.W_rec.tobytes()).hexdigest()[:8]
                else:
                    param_hash = "unknown"
                    
                cache_key = f"stability_{param_hash}"
                
                # Check cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Compute and cache
                result = stability_fn(model)
                self.cache.put(cache_key, result)
                return result
                
            except Exception:
                # If caching fails, just compute normally
                return stability_fn(model)
                
        return wrapper
    
    def precompute_common_operations(self, model, common_inputs: list):
        """Precompute and cache common operations."""
        for inputs, hidden_state in common_inputs:
            try:
                # Warm up cache with common inputs
                _ = model(inputs, hidden_state)
                
                # Cache stability measures
                _ = model.stability_measure()
                
            except Exception as e:
                warnings.warn(f"Failed to precompute for input: {e}")
    
    def get_cache_stats(self):
        """Get caching statistics."""
        return self.cache.get_stats()


class ModelCheckpointCache:
    """Cache for model checkpoints and trained parameters."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, model, epoch: int, metrics: Dict[str, float], 
                       optimizer_state=None):
        """Save model checkpoint."""
        checkpoint_data = {
            'model_state': model,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        if optimizer_state is not None:
            checkpoint_data['optimizer_state'] = optimizer_state
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Load model checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        try:
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint: {e}")
        return None
    
    def load_best_checkpoint(self, metric: str = 'val_loss', mode: str = 'min'):
        """Load best checkpoint based on metric."""
        best_checkpoint = None
        best_value = float('inf') if mode == 'min' else float('-inf')
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_epoch_*.pkl"):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    
                if metric in checkpoint['metrics']:
                    value = checkpoint['metrics'][metric]
                    if (mode == 'min' and value < best_value) or \
                       (mode == 'max' and value > best_value):
                        best_value = value
                        best_checkpoint = checkpoint
                        
            except Exception as e:
                warnings.warn(f"Failed to read checkpoint {checkpoint_file}: {e}")
        
        return best_checkpoint
    
    def clean_old_checkpoints(self, keep_last: int = 5):
        """Clean old checkpoints, keeping only the most recent ones."""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pkl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Remove old checkpoints
        for checkpoint_file in checkpoint_files[keep_last:]:
            checkpoint_file.unlink()


# Global cache instances
computation_cache = ComputationCache()
checkpoint_cache = ModelCheckpointCache()