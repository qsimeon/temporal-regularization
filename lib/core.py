"""
Core module for regularization over time in fluid learning neural networks.

This module provides the main components for implementing temporal regularization
in neural networks that adapt continuously over time. It includes regularizers,
fluid learning layers, and temporal constraint mechanisms.
"""

import numpy as np
from typing import Optional, Callable, List, Tuple, Dict, Any
import warnings


class TemporalRegularizer:
    """
    Implements regularization over time for fluid learning neural networks.
    
    This regularizer penalizes rapid changes in network parameters over time,
    encouraging smooth adaptation while allowing gradual learning.
    
    Attributes:
        lambda_reg (float): Regularization strength coefficient.
        decay_rate (float): Exponential decay rate for historical parameter importance.
        memory_length (int): Number of historical parameter states to maintain.
        regularization_type (str): Type of regularization ('l1', 'l2', or 'elastic').
    """
    
    def __init__(
        self,
        lambda_reg: float = 0.01,
        decay_rate: float = 0.95,
        memory_length: int = 10,
        regularization_type: str = 'l2'
    ):
        """
        Initialize the temporal regularizer.
        
        Args:
            lambda_reg: Regularization strength (higher = more penalty on changes).
            decay_rate: Decay factor for older parameter states (0-1).
            memory_length: Number of past parameter states to track.
            regularization_type: Type of norm to use ('l1', 'l2', or 'elastic').
        """
        self.lambda_reg = lambda_reg
        self.decay_rate = decay_rate
        self.memory_length = memory_length
        self.regularization_type = regularization_type
        self.parameter_history: List[np.ndarray] = []
        self.timestep = 0
        
    def compute_penalty(
        self,
        current_params: np.ndarray,
        reference_params: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute the temporal regularization penalty.
        
        Args:
            current_params: Current network parameters.
            reference_params: Optional reference parameters. If None, uses parameter history.
            
        Returns:
            Regularization penalty value.
        """
        if reference_params is None:
            if len(self.parameter_history) == 0:
                return 0.0
            reference_params = self._compute_weighted_reference()
        
        diff = current_params - reference_params
        
        if self.regularization_type == 'l1':
            penalty = np.sum(np.abs(diff))
        elif self.regularization_type == 'l2':
            penalty = np.sum(diff ** 2)
        elif self.regularization_type == 'elastic':
            penalty = 0.5 * np.sum(np.abs(diff)) + 0.5 * np.sum(diff ** 2)
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization_type}")
        
        return self.lambda_reg * penalty
    
    def compute_gradient(
        self,
        current_params: np.ndarray,
        reference_params: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the gradient of the regularization penalty.
        
        Args:
            current_params: Current network parameters.
            reference_params: Optional reference parameters.
            
        Returns:
            Gradient of the regularization penalty with respect to current_params.
        """
        if reference_params is None:
            if len(self.parameter_history) == 0:
                return np.zeros_like(current_params)
            reference_params = self._compute_weighted_reference()
        
        diff = current_params - reference_params
        
        if self.regularization_type == 'l1':
            grad = np.sign(diff)
        elif self.regularization_type == 'l2':
            grad = 2 * diff
        elif self.regularization_type == 'elastic':
            grad = 0.5 * np.sign(diff) + diff
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization_type}")
        
        return self.lambda_reg * grad
    
    def update_history(self, params: np.ndarray) -> None:
        """
        Update the parameter history with new parameters.
        
        Args:
            params: New parameter state to add to history.
        """
        self.parameter_history.append(params.copy())
        if len(self.parameter_history) > self.memory_length:
            self.parameter_history.pop(0)
        self.timestep += 1
    
    def _compute_weighted_reference(self) -> np.ndarray:
        """
        Compute weighted reference parameters from history.
        
        Returns:
            Weighted average of historical parameters.
        """
        if len(self.parameter_history) == 0:
            raise ValueError("No parameter history available")
        
        weights = np.array([
            self.decay_rate ** (len(self.parameter_history) - i - 1)
            for i in range(len(self.parameter_history))
        ])
        weights /= weights.sum()
        
        weighted_params = sum(
            w * p for w, p in zip(weights, self.parameter_history)
        )
        return weighted_params
    
    def reset(self) -> None:
        """Reset the regularizer state and clear history."""
        self.parameter_history.clear()
        self.timestep = 0


class FluidLayer:
    """
    A neural network layer with fluid learning capabilities.
    
    This layer can adapt its parameters over time while being constrained
    by temporal regularization to prevent catastrophic forgetting.
    
    Attributes:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output features.
        weights (np.ndarray): Layer weight matrix.
        bias (np.ndarray): Layer bias vector.
        activation (Callable): Activation function.
        regularizer (TemporalRegularizer): Temporal regularizer instance.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        regularizer: Optional[TemporalRegularizer] = None,
        init_scale: float = 0.01
    ):
        """
        Initialize a fluid learning layer.
        
        Args:
            input_dim: Number of input features.
            output_dim: Number of output features.
            activation: Activation function (default: None, linear).
            regularizer: Temporal regularizer instance.
            init_scale: Scale for weight initialization.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation if activation is not None else lambda x: x
        self.regularizer = regularizer
        
        # Initialize weights and bias
        self.weights = np.random.randn(input_dim, output_dim) * init_scale
        self.bias = np.zeros(output_dim)
        
        # Cache for backpropagation
        self._last_input: Optional[np.ndarray] = None
        self._last_preactivation: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            x: Input array of shape (batch_size, input_dim) or (input_dim,).
            
        Returns:
            Output array of shape (batch_size, output_dim) or (output_dim,).
        """
        self._last_input = x
        self._last_preactivation = x @ self.weights + self.bias
        return self.activation(self._last_preactivation)
    
    def get_parameters(self) -> np.ndarray:
        """
        Get flattened parameter vector.
        
        Returns:
            Flattened array of all layer parameters.
        """
        return np.concatenate([self.weights.flatten(), self.bias.flatten()])
    
    def set_parameters(self, params: np.ndarray) -> None:
        """
        Set layer parameters from flattened vector.
        
        Args:
            params: Flattened parameter array.
        """
        weight_size = self.input_dim * self.output_dim
        self.weights = params[:weight_size].reshape(self.input_dim, self.output_dim)
        self.bias = params[weight_size:]
    
    def update_with_regularization(self, grad_weights: np.ndarray, grad_bias: np.ndarray, learning_rate: float) -> float:
        """
        Update parameters with temporal regularization.
        
        Args:
            grad_weights: Gradient with respect to weights.
            grad_bias: Gradient with respect to bias.
            learning_rate: Learning rate for parameter update.
            
        Returns:
            Regularization penalty value.
        """
        penalty = 0.0
        
        if self.regularizer is not None:
            current_params = self.get_parameters()
            reg_grad = self.regularizer.compute_gradient(current_params)
            penalty = self.regularizer.compute_penalty(current_params)
            
            # Split regularization gradient
            weight_size = self.input_dim * self.output_dim
            reg_grad_weights = reg_grad[:weight_size].reshape(self.input_dim, self.output_dim)
            reg_grad_bias = reg_grad[weight_size:]
            
            # Update with combined gradients
            self.weights -= learning_rate * (grad_weights + reg_grad_weights)
            self.bias -= learning_rate * (grad_bias + reg_grad_bias)
            
            # Update history
            self.regularizer.update_history(self.get_parameters())
        else:
            # Standard update without regularization
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias
        
        return penalty


class AdaptiveTemporalRegularizer(TemporalRegularizer):
    """
    Adaptive temporal regularizer that adjusts regularization strength based on learning dynamics.
    
    This extends the base TemporalRegularizer with adaptive mechanisms that
    modulate the regularization strength based on observed parameter changes.
    """
    
    def __init__(
        self,
        lambda_reg: float = 0.01,
        decay_rate: float = 0.95,
        memory_length: int = 10,
        regularization_type: str = 'l2',
        adaptation_rate: float = 0.1,
        min_lambda: float = 0.001,
        max_lambda: float = 0.1
    ):
        """
        Initialize adaptive temporal regularizer.
        
        Args:
            lambda_reg: Initial regularization strength.
            decay_rate: Decay factor for older parameter states.
            memory_length: Number of past parameter states to track.
            regularization_type: Type of norm to use.
            adaptation_rate: Rate at which lambda adapts.
            min_lambda: Minimum regularization strength.
            max_lambda: Maximum regularization strength.
        """
        super().__init__(lambda_reg, decay_rate, memory_length, regularization_type)
        self.adaptation_rate = adaptation_rate
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.change_history: List[float] = []
    
    def adapt_lambda(self, current_params: np.ndarray) -> None:
        """
        Adapt the regularization strength based on parameter changes.
        
        Args:
            current_params: Current parameter state.
        """
        if len(self.parameter_history) > 0:
            last_params = self.parameter_history[-1]
            change_magnitude = np.linalg.norm(current_params - last_params)
            self.change_history.append(change_magnitude)
            
            if len(self.change_history) > self.memory_length:
                self.change_history.pop(0)
            
            # Increase lambda if changes are large, decrease if small
            if len(self.change_history) >= 2:
                avg_change = np.mean(self.change_history)
                if avg_change > np.median(self.change_history):
                    self.lambda_reg = min(self.max_lambda, self.lambda_reg * (1 + self.adaptation_rate))
                else:
                    self.lambda_reg = max(self.min_lambda, self.lambda_reg * (1 - self.adaptation_rate))
    
    def update_history(self, params: np.ndarray) -> None:
        """
        Update history and adapt regularization strength.
        
        Args:
            params: New parameter state to add to history.
        """
        self.adapt_lambda(params)
        super().update_history(params)
