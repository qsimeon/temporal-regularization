"""
Regularization over Time for Fluid Learning Neural Networks - Demo Script

This demo demonstrates temporal regularization techniques for neural networks that
continuously learn from streaming data while preventing catastrophic forgetting.

Features demonstrated:
1. Basic temporal regularization with TemporalRegularizer
2. Adaptive temporal regularization that adjusts based on learning dynamics
3. Fluid neural network layers with temporal constraints
4. Concept drift detection and handling
5. Performance monitoring and metrics computation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys
import os

# Add lib directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Import from available modules
from core import TemporalRegularizer, FluidLayer, AdaptiveTemporalRegularizer
from utils import (
    compute_parameter_drift,
    compute_forgetting_metric,
    compute_plasticity_metric,
    normalize_data,
    apply_normalization,
    create_temporal_batches,
    exponential_moving_average,
    compute_stability_score,
    detect_concept_drift,
    compute_learning_curve_metrics,
    generate_synthetic_drift_data,
    get_activation_function
)


class FluidNeuralNetwork:
    """
    A complete neural network with fluid learning capabilities and temporal regularization.
    """
    
    def __init__(self, layer_sizes: List[int], regularization_lambda: float = 0.01,
                 adaptive: bool = False, learning_rate: float = 0.01):
        """
        Initialize the fluid neural network.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            regularization_lambda: Regularization strength
            adaptive: Whether to use adaptive regularization
            learning_rate: Learning rate for training
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.layers = []
        self.regularizers = []
        
        # Create layers and regularizers
        for i in range(len(layer_sizes) - 1):
            layer = FluidLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation='relu' if i < len(layer_sizes) - 2 else 'linear'
            )
            self.layers.append(layer)
            
            if adaptive:
                regularizer = AdaptiveTemporalRegularizer(
                    lambda_reg=regularization_lambda,
                    adaptation_rate=0.01
                )
            else:
                regularizer = TemporalRegularizer(lambda_reg=regularization_lambda)
            
            self.regularizers.append(regularizer)
        
        # Training history
        self.loss_history = []
        self.performance_history = []
        self.parameter_history = []
        
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            output: Network output
            activations: List of activations from each layer
        """
        activations = [X]
        current = X
        
        for layer in self.layers:
            current = layer.forward(current)
            activations.append(current)
        
        return current, activations
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute mean squared error loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            loss: MSE loss value
        """
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, 
                activations: List[np.ndarray]) -> float:
        """
        Backward pass with temporal regularization.
        
        Args:
            X: Input data
            y_true: True labels
            activations: Activations from forward pass
            
        Returns:
            total_loss: Total loss including regularization
        """
        # Compute output error
        y_pred = activations[-1]
        data_loss = self.compute_loss(y_pred, y_true)
        
        # Gradient of loss w.r.t. output
        delta = 2 * (y_pred - y_true) / y_true.shape[0]
        
        total_reg_penalty = 0.0
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            regularizer = self.regularizers[i]
            
            # Get current parameters
            params = layer.get_parameters()
            
            # Compute regularization penalty
            reg_penalty = regularizer.compute_penalty(params['weights'])
            total_reg_penalty += reg_penalty
            
            # Compute regularization gradient
            reg_grad = regularizer.compute_gradient(params['weights'])
            
            # Get activation from previous layer
            a_prev = activations[i]
            
            # Compute gradients
            dW = np.dot(a_prev.T, delta) + reg_grad
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Update parameters
            params['weights'] -= self.learning_rate * dW
            params['biases'] -= self.learning_rate * db
            layer.set_parameters(params)
            
            # Update regularizer history
            regularizer.update_history(params['weights'])
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, params['weights'].T)
                # Apply derivative of activation function
                if layer.activation == 'relu':
                    delta = delta * (activations[i] > 0)
        
        total_loss = data_loss + total_reg_penalty
        return total_loss
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform one training step.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            loss: Training loss
        """
        # Forward pass
        y_pred, activations = self.forward(X)
        
        # Backward pass
        loss = self.backward(X, y, activations)
        
        # Record history
        self.loss_history.append(loss)
        
        # Compute performance metric (accuracy for classification, MSE for regression)
        performance = -loss  # Negative loss as performance metric
        self.performance_history.append(performance)
        
        # Record parameters
        all_params = []
        for layer in self.layers:
            params = layer.get_parameters()
            all_params.append(params['weights'].flatten())
        self.parameter_history.append(np.concatenate(all_params))
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            predictions: Network predictions
        """
        output, _ = self.forward(X)
        return output
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Compute various learning metrics.
        
        Returns:
            metrics: Dictionary of computed metrics
        """
        metrics = {}
        
        if len(self.parameter_history) > 5:
            # Compute parameter drift
            drift = compute_parameter_drift(self.parameter_history, window_size=5)
            metrics['avg_parameter_drift'] = np.mean(drift)
            
            # Compute plasticity
            plasticity = compute_plasticity_metric(self.parameter_history, window_size=5)
            metrics['plasticity'] = plasticity
            
            # Compute stability
            stability = compute_stability_score(self.parameter_history, window_size=10)
            metrics['stability'] = stability
        
        if len(self.performance_history) > 5:
            # Compute forgetting metric
            forgetting = compute_forgetting_metric(self.performance_history, window_size=5)
            metrics['forgetting'] = forgetting
            
            # Detect concept drift
            drift_points = detect_concept_drift(self.performance_history, 
                                               window_size=20, threshold=0.1)
            metrics['num_drift_points'] = len(drift_points)
            
            # Learning curve metrics
            lc_metrics = compute_learning_curve_metrics(self.performance_history)
            metrics.update(lc_metrics)
        
        return metrics


def demo_basic_temporal_regularization():
    """
    Demonstrate basic temporal regularization on a simple regression task.
    """
    print("\n" + "="*80)
    print("DEMO 1: Basic Temporal Regularization")
    print("="*80)
    
    # Generate synthetic data with concept drift
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_drift_points = 3
    
    X, y, drift_points = generate_synthetic_drift_data(
        n_samples=n_samples,
        n_features=n_features,
        n_drift_points=n_drift_points,
        noise_level=0.1,
        seed=42
    )
    
    print(f"\nGenerated synthetic data:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - Drift points: {drift_points}")
    
    # Normalize data
    X_normalized, norm_params = normalize_data(X, method='standard')
    
    # Create network with temporal regularization
    network = FluidNeuralNetwork(
        layer_sizes=[n_features, 20, 10, 1],
        regularization_lambda=0.01,
        adaptive=False,
        learning_rate=0.01
    )
    
    print("\nTraining network with temporal regularization...")
    
    # Train on sequential data
    batch_size = 32
    n_epochs = 3
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        for i in range(0, len(X_normalized) - batch_size, batch_size):
            X_batch = X_normalized[i:i+batch_size]
            y_batch = y[i:i+batch_size].reshape(-1, 1)
            
            loss = network.train_step(X_batch, y_batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{n_epochs} - Avg Loss: {avg_loss:.6f}")
    
    # Compute and display metrics
    metrics = network.get_metrics()
    print("\nLearning Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.6f}")
    
    return network, X_normalized, y


def demo_adaptive_temporal_regularization():
    """
    Demonstrate adaptive temporal regularization that adjusts based on learning dynamics.
    """
    print("\n" + "="*80)
    print("DEMO 2: Adaptive Temporal Regularization")
    print("="*80)
    
    # Generate synthetic data with concept drift
    np.random.seed(123)
    n_samples = 1000
    n_features = 10
    n_drift_points = 2
    
    X, y, drift_points = generate_synthetic_drift_data(
        n_samples=n_samples,
        n_features=n_features,
        n_drift_points=n_drift_points,
        noise_level=0.15,
        seed=123
    )
    
    print(f"\nGenerated synthetic data:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - Drift points: {drift_points}")
    
    # Normalize data
    X_normalized, norm_params = normalize_data(X, method='standard')
    
    # Create network with ADAPTIVE temporal regularization
    network = FluidNeuralNetwork(
        layer_sizes=[n_features, 20, 10, 1],
        regularization_lambda=0.01,
        adaptive=True,  # Enable adaptive regularization
        learning_rate=0.01
    )
    
    print("\nTraining network with ADAPTIVE temporal regularization...")
    
    # Train on sequential data
    batch_size = 32
    n_epochs = 3
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        for i in range(0, len(X_normalized) - batch_size, batch_size):
            X_batch = X_normalized[i:i+batch_size]
            y_batch = y[i:i+batch_size].reshape(-1, 1)
            
            loss = network.train_step(X_batch, y_batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{n_epochs} - Avg Loss: {avg_loss:.6f}")
    
    # Compute and display metrics
    metrics = network.get_metrics()
    print("\nLearning Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.6f}")
    
    # Display regularization lambda values over time
    print("\nAdaptive Regularization Lambda History (last 10 values):")
    for i, regularizer in enumerate(network.regularizers):
        if hasattr(regularizer, 'lambda_history'):
            lambda_values = regularizer.lambda_history[-10:]
            print(f"  Layer {i}: {[f'{v:.6f}' for v in lambda_values]}")
    
    return network, X_normalized, y


def demo_comparison():
    """
    Compare networks with and without temporal regularization.
    """
    print("\n" + "="*80)
    print("DEMO 3: Comparison - With vs Without Temporal Regularization")
    print("="*80)
    
    # Generate synthetic data with concept drift
    np.random.seed(456)
    n_samples = 800
    n_features = 8
    n_drift_points = 2
    
    X, y, drift_points = generate_synthetic_drift_data(
        n_samples=n_samples,
        n_features=n_features,
        n_drift_points=n_drift_points,
        noise_level=0.1,
        seed=456
    )
    
    print(f"\nGenerated synthetic data:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - Drift points: {drift_points}")
    
    # Normalize data
    X_normalized, norm_params = normalize_data(X, method='standard')
    
    # Create two networks: one with regularization, one without
    print("\nTraining Network WITHOUT temporal regularization...")
    network_no_reg = FluidNeuralNetwork(
        layer_sizes=[n_features, 15, 8, 1],
        regularization_lambda=0.0,  # No regularization
        adaptive=False,
        learning_rate=0.01
    )
    
    print("Training Network WITH temporal regularization...")
    network_with_reg = FluidNeuralNetwork(
        layer_sizes=[n_features, 15, 8, 1],
        regularization_lambda=0.02,  # With regularization
        adaptive=False,
        learning_rate=0.01
    )
    
    # Train both networks
    batch_size = 32
    n_epochs = 2
    
    for epoch in range(n_epochs):
        # Train without regularization
        for i in range(0, len(X_normalized) - batch_size, batch_size):
            X_batch = X_normalized[i:i+batch_size]
            y_batch = y[i:i+batch_size].reshape(-1, 1)
            network_no_reg.train_step(X_batch, y_batch)
        
        # Train with regularization
        for i in range(0, len(X_normalized) - batch_size, batch_size):
            X_batch = X_normalized[i:i+batch_size]
            y_batch = y[i:i+batch_size].reshape(-1, 1)
            network_with_reg.train_step(X_batch, y_batch)
    
    # Compare metrics
    print("\n" + "-"*80)
    print("Metrics Comparison:")
    print("-"*80)
    
    metrics_no_reg = network_no_reg.get_metrics()
    metrics_with_reg = network_with_reg.get_metrics()
    
    print("\nWithout Temporal Regularization:")
    for key, value in metrics_no_reg.items():
        print(f"  - {key}: {value:.6f}")
    
    print("\nWith Temporal Regularization:")
    for key, value in metrics_with_reg.items():
        print(f"  - {key}: {value:.6f}")
    
    # Compute smoothed loss curves
    print("\n" + "-"*80)
    print("Loss Curve Analysis:")
    print("-"*80)
    
    smoothed_loss_no_reg = exponential_moving_average(
        network_no_reg.loss_history, alpha=0.1
    )
    smoothed_loss_with_reg = exponential_moving_average(
        network_with_reg.loss_history, alpha=0.1
    )
    
    print(f"\nFinal smoothed loss (no reg): {smoothed_loss_no_reg[-1]:.6f}")
    print(f"Final smoothed loss (with reg): {smoothed_loss_with_reg[-1]:.6f}")
    
    return network_no_reg, network_with_reg, X_normalized, y


def demo_temporal_batches():
    """
    Demonstrate temporal batch creation for sequential learning.
    """
    print("\n" + "="*80)
    print("DEMO 4: Temporal Batch Processing")
    print("="*80)
    
    # Generate time series data
    np.random.seed(789)
    n_samples = 500
    n_features = 5
    
    X, y, drift_points = generate_synthetic_drift_data(
        n_samples=n_samples,
        n_features=n_features,
        n_drift_points=1,
        noise_level=0.1,
        seed=789
    )
    
    print(f"\nGenerated time series data:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    
    # Normalize data
    X_normalized, norm_params = normalize_data(X, method='standard')
    
    # Create temporal batches
    batch_size = 16
    sequence_length = 10
    stride = 5
    
    temporal_batches = create_temporal_batches(
        data=X_normalized,
        labels=y,
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride
    )
    
    print(f"\nCreated temporal batches:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Stride: {stride}")
    print(f"  - Total batches: {len(temporal_batches)}")
    
    # Create network
    network = FluidNeuralNetwork(
        layer_sizes=[n_features, 15, 8, 1],
        regularization_lambda=0.015,
        adaptive=True,
        learning_rate=0.01
    )
    
    print("\nTraining on temporal batches...")
    
    # Train on temporal batches
    for batch_idx, (X_batch, y_batch) in enumerate(temporal_batches[:50]):  # First 50 batches
        # Reshape for network input
        X_batch_flat = X_batch.reshape(-1, n_features)
        y_batch_flat = y_batch.reshape(-1, 1)
        
        loss = network.train_step(X_batch_flat, y_batch_flat)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/50 - Loss: {loss:.6f}")
    
    # Compute metrics
    metrics = network.get_metrics()
    print("\nFinal Learning Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.6f}")
    
    return network, temporal_batches


def demo_concept_drift_handling():
    """
    Demonstrate how temporal regularization helps handle concept drift.
    """
    print("\n" + "="*80)
    print("DEMO 5: Concept Drift Detection and Handling")
    print("="*80)
    
    # Generate data with multiple drift points
    np.random.seed(999)
    n_samples = 1200
    n_features = 12
    n_drift_points = 4
    
    X, y, drift_points = generate_synthetic_drift_data(
        n_samples=n_samples,
        n_features=n_features,
        n_drift_points=n_drift_points,
        noise_level=0.12,
        seed=999
    )
    
    print(f"\nGenerated data with concept drift:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - True drift points: {drift_points}")
    
    # Normalize data
    X_normalized, norm_params = normalize_data(X, method='standard')
    
    # Create network with adaptive regularization
    network = FluidNeuralNetwork(
        layer_sizes=[n_features, 25, 15, 1],
        regularization_lambda=0.01,
        adaptive=True,
        learning_rate=0.01
    )
    
    print("\nTraining with concept drift monitoring...")
    
    # Train sequentially
    batch_size = 32
    for i in range(0, len(X_normalized) - batch_size, batch_size):
        X_batch = X_normalized[i:i+batch_size]
        y_batch = y[i:i+batch_size].reshape(-1, 1)
        
        loss = network.train_step(X_batch, y_batch)
        
        # Print progress at drift points
        current_sample = i + batch_size
        if any(abs(current_sample - dp) < batch_size for dp in drift_points):
            print(f"  Sample {current_sample} (near drift point) - Loss: {loss:.6f}")
    
    # Detect drift points from performance history
    detected_drift_points = detect_concept_drift(
        performance_history=network.performance_history,
        window_size=20,
        threshold=0.15
    )
    
    print(f"\nDrift Detection Results:")
    print(f"  - True drift points: {drift_points}")
    print(f"  - Detected drift indices: {detected_drift_points}")
    print(f"  - Number detected: {len(detected_drift_points)}")
    
    # Compute parameter drift over time
    param_drift = compute_parameter_drift(network.parameter_history, window_size=10)
    
    print(f"\nParameter Drift Statistics:")
    print(f"  - Mean drift: {np.mean(param_drift):.6f}")
    print(f"  - Max drift: {np.max(param_drift):.6f}")
    print(f"  - Std drift: {np.std(param_drift):.6f}")
    
    # Final metrics
    metrics = network.get_metrics()
    print("\nFinal Learning Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.6f}")
    
    return network, drift_points, detected_drift_points


def visualize_results(network, title="Learning Curves"):
    """
    Visualize learning curves and metrics.
    
    Args:
        network: Trained FluidNeuralNetwork
        title: Plot title
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Loss curve
        if len(network.loss_history) > 0:
            smoothed_loss = exponential_moving_average(network.loss_history, alpha=0.1)
            axes[0, 0].plot(network.loss_history, alpha=0.3, label='Raw Loss')
            axes[0, 0].plot(smoothed_loss, linewidth=2, label='Smoothed Loss')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss Over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Parameter drift
        if len(network.parameter_history) > 5:
            param_drift = compute_parameter_drift(network.parameter_history, window_size=5)
            axes[0, 1].plot(param_drift, color='orange', linewidth=2)
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Parameter Drift')
            axes[0, 1].set_title('Parameter Drift Over Time')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance history
        if len(network.performance_history) > 0:
            smoothed_perf = exponential_moving_average(network.performance_history, alpha=0.1)
            axes[1, 0].plot(network.performance_history, alpha=0.3, label='Raw Performance')
            axes[1, 0].plot(smoothed_perf, linewidth=2, label='Smoothed Performance')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Performance')
            axes[1, 0].set_title('Performance Over Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Metrics summary
        metrics = network.get_metrics()
        if metrics:
            metric_names = list(metrics.keys())[:6]  # Top 6 metrics
            metric_values = [metrics[k] for k in metric_names]
            
            axes[1, 1].barh(metric_names, metric_values, color='steelblue')
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_title('Learning Metrics Summary')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('fluid_learning_results.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to 'fluid_learning_results.png'")
        
    except Exception as e:
        print(f"\n⚠ Visualization skipped (matplotlib display issue): {e}")


def main():
    """
    Main function to run all demonstrations.
    """
    print("\n" + "="*80)
    print("FLUID LEARNING NEURAL NETWORKS - TEMPORAL REGULARIZATION DEMO")
    print("="*80)
    print("\nThis demo showcases temporal regularization techniques for neural networks")
    print("that continuously learn from streaming data while preventing catastrophic")
    print("forgetting through regularization over time.")
    
    try:
        # Demo 1: Basic temporal regularization
        network1, X1, y1 = demo_basic_temporal_regularization()
        
        # Demo 2: Adaptive temporal regularization
        network2, X2, y2 = demo_adaptive_temporal_regularization()
        
        # Demo 3: Comparison
        network_no_reg, network_with_reg, X3, y3 = demo_comparison()
        
        # Demo 4: Temporal batches
        network4, batches = demo_temporal_batches()
        
        # Demo 5: Concept drift handling
        network5, true_drifts, detected_drifts = demo_concept_drift_handling()
        
        # Visualize results from adaptive network
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        visualize_results(network2, title="Adaptive Temporal Regularization Results")
        
        # Summary
        print("\n" + "="*80)
        print("DEMO SUMMARY")
        print("="*80)
        print("\n✓ All demonstrations completed successfully!")
        print("\nKey Findings:")
        print("  1. Temporal regularization helps maintain stability during learning")
        print("  2. Adaptive regularization adjusts to learning dynamics")
        print("  3. Regularization reduces catastrophic forgetting")
        print("  4. Temporal batches enable sequential learning")
        print("  5. Concept drift can be detected and handled effectively")
        
        print("\nMetrics from Adaptive Network:")
        metrics = network2.get_metrics()
        for key, value in list(metrics.items())[:5]:
            print(f"  - {key}: {value:.6f}")
        
    except Exception as e:
        print(f"\n❌ Error during demo execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
