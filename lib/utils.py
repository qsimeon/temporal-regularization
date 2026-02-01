"""
Utility functions for fluid learning neural networks with temporal regularization.

This module provides helper functions for data processing, visualization,
metrics computation, and common operations used in fluid learning systems.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import warnings


def compute_parameter_drift(
    params_history: List[np.ndarray],
    window_size: Optional[int] = None
) -> np.ndarray:
    """
    Compute the drift (change magnitude) of parameters over time.
    
    Args:
        params_history: List of parameter arrays over time.
        window_size: Optional window size for computing drift. If None, uses consecutive steps.
        
    Returns:
        Array of drift magnitudes at each timestep.
    """
    if len(params_history) < 2:
        return np.array([])
    
    if window_size is None:
        window_size = 1
    
    drifts = []
    for i in range(window_size, len(params_history)):
        diff = params_history[i] - params_history[i - window_size]
        drift = np.linalg.norm(diff)
        drifts.append(drift)
    
    return np.array(drifts)


def compute_forgetting_metric(
    performance_history: List[float],
    window_size: int = 5
) -> float:
    """
    Compute a forgetting metric based on performance degradation.
    
    Args:
        performance_history: List of performance scores over time (higher is better).
        window_size: Window size for comparing past vs recent performance.
        
    Returns:
        Forgetting score (higher means more forgetting).
    """
    if len(performance_history) < 2 * window_size:
        return 0.0
    
    past_performance = np.mean(performance_history[-2*window_size:-window_size])
    recent_performance = np.mean(performance_history[-window_size:])
    
    forgetting = max(0, past_performance - recent_performance)
    return forgetting


def compute_plasticity_metric(
    params_history: List[np.ndarray],
    window_size: int = 5
) -> float:
    """
    Compute a plasticity metric based on parameter change rate.
    
    Args:
        params_history: List of parameter arrays over time.
        window_size: Window size for computing plasticity.
        
    Returns:
        Plasticity score (higher means more plastic/adaptive).
    """
    if len(params_history) < window_size + 1:
        return 0.0
    
    recent_params = params_history[-window_size:]
    changes = []
    
    for i in range(1, len(recent_params)):
        change = np.linalg.norm(recent_params[i] - recent_params[i-1])
        changes.append(change)
    
    return np.mean(changes) if changes else 0.0


def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
    axis: int = 0,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data using various methods.
    
    Args:
        data: Input data array.
        method: Normalization method ('standard', 'minmax', or 'robust').
        axis: Axis along which to normalize.
        epsilon: Small constant for numerical stability.
        
    Returns:
        Tuple of (normalized_data, normalization_params).
    """
    if method == 'standard':
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True) + epsilon
        normalized = (data - mean) / std
        params = {'mean': mean, 'std': std, 'method': 'standard'}
    
    elif method == 'minmax':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + epsilon)
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    
    elif method == 'robust':
        median = np.median(data, axis=axis, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis=axis, keepdims=True)
        iqr = q75 - q25 + epsilon
        normalized = (data - median) / iqr
        params = {'median': median, 'iqr': iqr, 'method': 'robust'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def apply_normalization(
    data: np.ndarray,
    normalization_params: Dict[str, Any]
) -> np.ndarray:
    """
    Apply previously computed normalization parameters to new data.
    
    Args:
        data: Input data to normalize.
        normalization_params: Parameters from normalize_data function.
        
    Returns:
        Normalized data array.
    """
    method = normalization_params['method']
    
    if method == 'standard':
        return (data - normalization_params['mean']) / normalization_params['std']
    elif method == 'minmax':
        min_val = normalization_params['min']
        max_val = normalization_params['max']
        return (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        return (data - normalization_params['median']) / normalization_params['iqr']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_temporal_batches(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    sequence_length: int,
    stride: int = 1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create temporal batches from sequential data.
    
    Args:
        data: Input data array of shape (n_samples, n_features).
        labels: Label array of shape (n_samples,) or (n_samples, n_outputs).
        batch_size: Number of sequences per batch.
        sequence_length: Length of each sequence.
        stride: Stride between consecutive sequences.
        
    Returns:
        List of (batch_data, batch_labels) tuples.
    """
    n_samples = len(data)
    batches = []
    
    # Generate sequence start indices
    start_indices = list(range(0, n_samples - sequence_length + 1, stride))
    
    # Create batches
    for i in range(0, len(start_indices), batch_size):
        batch_starts = start_indices[i:i + batch_size]
        
        batch_sequences = []
        batch_labels_list = []
        
        for start in batch_starts:
            seq = data[start:start + sequence_length]
            label = labels[start + sequence_length - 1]  # Use last label in sequence
            batch_sequences.append(seq)
            batch_labels_list.append(label)
        
        batch_data = np.array(batch_sequences)
        batch_labels_arr = np.array(batch_labels_list)
        batches.append((batch_data, batch_labels_arr))
    
    return batches


def exponential_moving_average(
    values: List[float],
    alpha: float = 0.1
) -> List[float]:
    """
    Compute exponential moving average of a sequence.
    
    Args:
        values: List of values.
        alpha: Smoothing factor (0 < alpha <= 1).
        
    Returns:
        List of smoothed values.
    """
    if not values:
        return []
    
    ema = [values[0]]
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    
    return ema


def compute_stability_score(
    params_history: List[np.ndarray],
    window_size: int = 10
) -> float:
    """
    Compute a stability score based on parameter variance.
    
    Args:
        params_history: List of parameter arrays over time.
        window_size: Window size for computing stability.
        
    Returns:
        Stability score (higher means more stable).
    """
    if len(params_history) < window_size:
        return 1.0
    
    recent_params = params_history[-window_size:]
    param_matrix = np.array(recent_params)
    variance = np.var(param_matrix, axis=0).mean()
    
    # Convert variance to stability (inverse relationship)
    stability = 1.0 / (1.0 + variance)
    return stability


def detect_concept_drift(
    performance_history: List[float],
    window_size: int = 20,
    threshold: float = 0.1
) -> List[int]:
    """
    Detect concept drift points based on performance changes.
    
    Args:
        performance_history: List of performance scores over time.
        window_size: Window size for drift detection.
        threshold: Threshold for detecting significant drift.
        
    Returns:
        List of indices where concept drift was detected.
    """
    if len(performance_history) < 2 * window_size:
        return []
    
    drift_points = []
    
    for i in range(window_size, len(performance_history) - window_size):
        before_window = performance_history[i - window_size:i]
        after_window = performance_history[i:i + window_size]
        
        mean_before = np.mean(before_window)
        mean_after = np.mean(after_window)
        
        # Detect significant drop in performance
        if mean_before - mean_after > threshold:
            drift_points.append(i)
    
    return drift_points


def compute_learning_curve_metrics(
    performance_history: List[float]
) -> Dict[str, float]:
    """
    Compute various metrics from a learning curve.
    
    Args:
        performance_history: List of performance scores over time.
        
    Returns:
        Dictionary containing learning curve metrics.
    """
    if not performance_history:
        return {}
    
    metrics = {
        'initial_performance': performance_history[0],
        'final_performance': performance_history[-1],
        'max_performance': max(performance_history),
        'min_performance': min(performance_history),
        'mean_performance': np.mean(performance_history),
        'std_performance': np.std(performance_history),
        'improvement': performance_history[-1] - performance_history[0]
    }
    
    # Compute convergence speed (timesteps to reach 90% of max)
    max_perf = max(performance_history)
    threshold = 0.9 * max_perf
    convergence_step = next((i for i, p in enumerate(performance_history) if p >= threshold), len(performance_history))
    metrics['convergence_speed'] = convergence_step
    
    return metrics


def generate_synthetic_drift_data(
    n_samples: int,
    n_features: int,
    n_drift_points: int = 2,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate synthetic data with concept drift for testing.
    
    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        n_drift_points: Number of concept drift points.
        noise_level: Level of noise to add.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (data, labels, drift_point_indices).
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = np.random.randn(n_samples, n_features)
    labels = np.zeros(n_samples)
    
    # Determine drift points
    drift_points = sorted(np.random.choice(
        range(n_samples // 4, 3 * n_samples // 4),
        size=n_drift_points,
        replace=False
    ).tolist())
    drift_points = [0] + drift_points + [n_samples]
    
    # Generate different concepts for each segment
    for i in range(len(drift_points) - 1):
        start, end = drift_points[i], drift_points[i + 1]
        
        # Random linear combination for this concept
        weights = np.random.randn(n_features)
        labels[start:end] = (data[start:end] @ weights > 0).astype(float)
    
    # Add noise
    noise = np.random.randn(n_samples, n_features) * noise_level
    data += noise
    
    return data, labels, drift_points[1:-1]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def _tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)


def get_activation_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function ('sigmoid', 'relu', 'tanh', 'linear').
        
    Returns:
        Activation function.
    """
    activations = {
        'sigmoid': _sigmoid,
        'relu': _relu,
        'tanh': _tanh,
        'linear': lambda x: x
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}. Choose from {list(activations.keys())}")
    
    return activations[name]
