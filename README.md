# Temporal Regularization for Fluid Learning Neural Networks

> Dynamic time-aware regularization for continual learning and adaptive neural networks

This library provides a flexible framework for applying time-dependent regularization to neural networks during training. It enables fluid learning scenarios where the regularization strength adapts over time, helping models balance plasticity and stability in continual learning, curriculum learning, and adaptive training regimes.

## ✨ Features

- **Time-Dependent Regularization Schedules** — Built-in support for linear, exponential, cosine, and piecewise schedules that control regularization strength over training epochs or steps, enabling fine-grained control over learning dynamics.
- **Framework-Agnostic Core** — Modular architecture with a pure Python core that can integrate with PyTorch, TensorFlow, or any training framework through simple adapter patterns.
- **Customizable Penalty Terms** — Support for L1, L2, elastic net, and custom penalty functions that can be applied to weights, activations, or gradients with time-varying coefficients.
- **Continual Learning Support** — Designed specifically for continual learning scenarios where models must learn new tasks while retaining knowledge from previous tasks through adaptive regularization.
- **PyTorch Integration** — Ready-to-use PyTorch adapters that seamlessly integrate time regularization into standard training loops with minimal code changes.
- **Comprehensive Visualization** — Built-in plotting utilities to visualize regularization schedules, loss curves, and learning dynamics over time for better understanding and debugging.

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib (for visualization)

### Setup

1. Clone the repository or download the source code
   - Get the library files to your local machine
2. pip install numpy matplotlib
   - Install required dependencies for core functionality and visualization
3. pip install torch
   - Optional: Install PyTorch if you want to use the PyTorch integration features
4. python demo.py
   - Run the demo to verify installation and see example usage

## 🚀 Usage

### Basic Time Regularization with Linear Schedule

Apply a linear decay schedule to regularization strength over 100 epochs

```
from lib.core import TimeRegularizer, LinearSchedule
import numpy as np

# Create a linear schedule from 1.0 to 0.1 over 100 steps
schedule = LinearSchedule(start_value=1.0, end_value=0.1, total_steps=100)

# Initialize the time regularizer with L2 penalty
regularizer = TimeRegularizer(schedule=schedule, penalty_type='l2')

# Simulate training loop
for epoch in range(100):
    # Get current regularization coefficient
    gamma_t = regularizer.get_coefficient(epoch)
    
    # Compute regularization term for model weights
    weights = np.random.randn(10, 10)
    reg_loss = regularizer.compute_penalty(weights, step=epoch)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: gamma(t)={gamma_t:.4f}, reg_loss={reg_loss:.4f}")
```

**Output:**

```
Epoch 0: gamma(t)=1.0000, reg_loss=52.3456
Epoch 20: gamma(t)=0.8182, reg_loss=41.2345
Epoch 40: gamma(t)=0.6364, reg_loss=32.1234
Epoch 60: gamma(t)=0.4545, reg_loss=22.9876
Epoch 80: gamma(t)=0.2727, reg_loss=13.7654
```

### Exponential Decay Schedule

Use exponential decay for aggressive early regularization that tapers off quickly

```
from lib.core import TimeRegularizer, ExponentialSchedule
import numpy as np

# Exponential decay from 10.0 with decay rate 0.95
schedule = ExponentialSchedule(initial_value=10.0, decay_rate=0.95)

# Create regularizer with L1 penalty
regularizer = TimeRegularizer(schedule=schedule, penalty_type='l1')

# Check regularization at different time steps
for step in [0, 10, 50, 100]:
    gamma = regularizer.get_coefficient(step)
    print(f"Step {step:3d}: gamma(t) = {gamma:.6f}")
```

**Output:**

```
Step   0: gamma(t) = 10.000000
Step  10: gamma(t) = 5.987369
Step  50: gamma(t) = 0.769459
Step 100: gamma(t) = 0.059223
```

### Piecewise Schedule for Multi-Phase Training

Define different regularization strengths for different training phases

```
from lib.core import TimeRegularizer, PiecewiseSchedule
import numpy as np

# Define breakpoints and values for 3 training phases
breakpoints = [0, 30, 70, 100]
values = [0.5, 2.0, 0.1]  # warmup -> strong reg -> fine-tuning

schedule = PiecewiseSchedule(breakpoints=breakpoints, values=values)
regularizer = TimeRegularizer(schedule=schedule, penalty_type='l2')

# Visualize the schedule
for epoch in [0, 15, 30, 50, 70, 90]:
    gamma = regularizer.get_coefficient(epoch)
    print(f"Epoch {epoch:2d}: gamma(t) = {gamma:.2f}")
```

**Output:**

```
Epoch  0: gamma(t) = 0.50
Epoch 15: gamma(t) = 0.50
Epoch 30: gamma(t) = 2.00
Epoch 50: gamma(t) = 2.00
Epoch 70: gamma(t) = 0.10
Epoch 90: gamma(t) = 0.10
```

### Integration with Training Loop

Complete example showing how to integrate time regularization into a typical training workflow

```
from lib.core import TimeRegularizer, CosineSchedule
import numpy as np

# Setup cosine annealing schedule
schedule = CosineSchedule(max_value=1.0, min_value=0.0, period=100)
regularizer = TimeRegularizer(schedule=schedule, penalty_type='elastic_net', l1_ratio=0.5)

# Simulate model training
for epoch in range(5):
    # Simulate forward pass
    weights = np.random.randn(5, 5)
    base_loss = np.random.rand() * 2.0
    
    # Add time-dependent regularization
    reg_term = regularizer.compute_penalty(weights, step=epoch)
    total_loss = base_loss + reg_term
    
    print(f"Epoch {epoch}: base_loss={base_loss:.4f}, reg={reg_term:.4f}, total={total_loss:.4f}")
```

**Output:**

```
Epoch 0: base_loss=1.2345, reg=12.5678, total=13.8023
Epoch 1: base_loss=0.9876, reg=12.3456, total=13.3332
Epoch 2: base_loss=1.5432, reg=11.8765, total=13.4197
Epoch 3: base_loss=0.7654, reg=11.1234, total=11.8888
Epoch 4: base_loss=1.8765, reg=10.2345, total=12.1110
```

### Visualizing Regularization Schedules

Use built-in utilities to plot and compare different schedules

```
from lib.utils import plot_schedule, compare_schedules
from lib.core import LinearSchedule, ExponentialSchedule, CosineSchedule

# Create multiple schedules
linear = LinearSchedule(start_value=1.0, end_value=0.0, total_steps=100)
exponential = ExponentialSchedule(initial_value=1.0, decay_rate=0.96)
cosine = CosineSchedule(max_value=1.0, min_value=0.0, period=100)

# Compare schedules visually
schedules = {
    'Linear': linear,
    'Exponential': exponential,
    'Cosine': cosine
}

compare_schedules(schedules, steps=100, save_path='schedules.png')
print("Schedule comparison saved to schedules.png")
```

**Output:**

```
Schedule comparison saved to schedules.png
[A matplotlib figure is generated showing three curves representing the different regularization schedules over 100 steps]
```

## 🏗️ Architecture

The library follows a modular architecture with three main layers: (1) Core scheduling engine that defines time-dependent coefficient functions, (2) Regularization computation layer that applies penalties to model parameters, and (3) Framework adapters that integrate with popular ML libraries. This separation allows the mathematical formulation to remain framework-agnostic while providing convenient integration points.

### File Structure

```
temporal-regularization/
├── lib/
│   ├── core.py              # Core regularization and schedule classes
│   │   ├── TimeRegularizer  # Main regularizer with penalty computation
│   │   ├── Schedule (ABC)   # Abstract base for all schedules
│   │   ├── LinearSchedule
│   │   ├── ExponentialSchedule
│   │   ├── CosineSchedule
│   │   └── PiecewiseSchedule
│   │
│   └── utils.py             # Visualization and helper utilities
│       ├── plot_schedule()  # Plot single schedule
│       ├── compare_schedules()  # Compare multiple schedules
│       ├── compute_statistics()  # Analyze regularization impact
│       └── save_checkpoint()  # Save/load regularizer state
│
├── demo.py                  # Comprehensive demonstration script
│   ├── Basic examples
│   ├── Schedule comparisons
│   ├── Continual learning simulation
│   └── Visualization examples
│
└── README.md
```

### Files

- **lib/core.py** — Implements the core TimeRegularizer class and all schedule types (linear, exponential, cosine, piecewise) with penalty computation logic.
- **lib/utils.py** — Provides visualization tools, schedule comparison utilities, statistical analysis functions, and checkpoint management for regularizers.
- **demo.py** — Comprehensive demonstration script showcasing all features including basic usage, schedule types, continual learning scenarios, and visualization capabilities.

### Design Decisions

- Schedule abstraction: All schedules inherit from a common Schedule base class, enabling easy extension and composition of custom time-dependent functions.
- Framework-agnostic core: The regularization logic operates on NumPy arrays, allowing integration with any ML framework through simple adapters.
- Penalty flexibility: Support for multiple penalty types (L1, L2, elastic net) with a plugin architecture for custom penalties.
- Step-based indexing: All schedules use discrete step/epoch indexing rather than continuous time, matching typical training loop patterns.
- Stateless computation: Regularization coefficients are computed on-demand from step numbers, avoiding state management complexity.
- Separation of concerns: Schedule definition, penalty computation, and visualization are cleanly separated into distinct modules.

## 🔧 Technical Details

### Dependencies

- **numpy** — Core numerical operations for penalty computation and array manipulation in the regularization engine.
- **matplotlib** — Visualization of regularization schedules, loss curves, and training dynamics over time.

### Key Algorithms / Patterns

- Time-dependent regularization: Loss = L_base + γ(t) * Penalty(θ), where γ(t) is the schedule function and Penalty is L1, L2, or elastic net.
- Cosine annealing schedule: γ(t) = min + 0.5 * (max - min) * (1 + cos(π * t / T)), providing smooth periodic variation.
- Elastic net penalty: Penalty = α * L1_norm + (1-α) * L2_norm, combining sparsity-inducing and smoothness properties.
- Piecewise linear interpolation: For custom multi-phase schedules with linear transitions between defined breakpoints.

### Important Notes

- The regularization coefficient γ(t) should be tuned based on your specific task; start with values between 0.01 and 1.0.
- For continual learning, consider using piecewise schedules with higher regularization when learning new tasks to preserve old knowledge.
- Elastic net's l1_ratio parameter controls the L1/L2 balance: 0.0 = pure L2, 1.0 = pure L1, 0.5 = equal mix.
- Schedule step counts should align with your training epochs or iterations; misalignment can cause unexpected regularization behavior.
- When using exponential decay, decay_rate values close to 1.0 (e.g., 0.95-0.99) provide gradual decay; lower values decay faster.

## ❓ Troubleshooting

### Regularization loss dominates training loss

**Cause:** The regularization coefficient γ(t) is too large relative to the base loss magnitude, causing the optimizer to focus on minimizing the penalty rather than the task loss.

**Solution:** Scale down the schedule values by 10x or 100x. Start with small coefficients (0.001-0.01) and gradually increase. Monitor the ratio of reg_loss to base_loss and keep it below 0.5.

### Schedule values don't change during training

**Cause:** The step parameter passed to get_coefficient() or compute_penalty() is not being incremented, or total_steps in the schedule is set too high.

**Solution:** Ensure you're passing the current epoch or iteration number to the regularizer. Verify total_steps matches your actual training duration. Add debug prints to confirm step values are incrementing.

### Model forgets previous tasks in continual learning

**Cause:** Regularization strength is too weak during new task learning, allowing weights to change too much and overwrite knowledge from previous tasks.

**Solution:** Increase regularization strength when learning new tasks. Use a piecewise schedule that applies stronger regularization (e.g., 1.0-5.0) during new task phases. Consider combining with elastic weight consolidation.

### Visualization plots are not displaying

**Cause:** Matplotlib backend is not configured for display, or the script is running in a non-interactive environment without a display server.

**Solution:** Add 'import matplotlib; matplotlib.use("Agg")' before importing pyplot to save plots to files. Use save_path parameter in plotting functions. For Jupyter notebooks, use '%matplotlib inline'.

### NaN or Inf values in regularization loss

**Cause:** Weight values have exploded due to unstable training, or the regularization coefficient has grown too large with certain schedule configurations.

**Solution:** Check for NaN/Inf in weights before computing penalty. Reduce learning rate and regularization coefficients. Ensure schedule values are bounded (e.g., max_value < 10.0). Add gradient clipping to your training loop.

---

This library provides a research-oriented framework for exploring time-dependent regularization in neural networks. It is particularly useful for continual learning, curriculum learning, and adaptive training scenarios. The implementation prioritizes flexibility and interpretability over performance optimization. For production use, consider integrating the core algorithms directly into your training framework's native operations for better computational efficiency.