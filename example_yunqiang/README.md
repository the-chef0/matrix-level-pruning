# Transform-Centric Layer Pruning

This repository implements a transform-centric approach to neural network pruning, focusing on maintaining the structural integrity of the network while removing redundant layers.

## Transform-Centric Principles

### 1. Every Group MUST Start with a Transform Layer

- Groups are anchored by Conv/Linear layers
- Non-transform layers can only be included if they depend on the transform
- Each group has a lead_transform that defines it

### 2. Clear Group Boundaries

- Groups end when the next transform layer is encountered
- Transform layers define natural pruning boundaries
- No group can span multiple transform layers

### 3. Dependency-Based Inclusion

Group Formation Rules:
```
Conv2d (lead) → BatchNorm2d → ReLU → Dropout
   ↑              ↑            ↑       ↑
   START          included     included included
   
Linear (lead) → ReLU
   ↑             ↑
   START         included
```

## 📊 Example Output

```
PRUNING GROUPS (each starting from a transform layer):
--------------------------------------------------------------------------------

Group 0: ✓ SAFE TO REMOVE
  Lead Transform: conv2a (Conv2d)
  Type: parallel_branch
  Impact Score: 0.125
  Parameters: 36,928 (8.5%)
  Layers (3):
    ★ conv2a: Conv2d (transform)
    → bn2a: BatchNorm2d (normalization)
    → relu2a: ReLU (activation)

Group 1: ✓ SAFE TO REMOVE
  Lead Transform: conv2b (Conv2d)
  Type: parallel_branch
  Impact Score: 0.156
  Parameters: 51,264 (11.8%)
  Layers (2):
    ★ conv2b: Conv2d (transform)
    → bn2b: BatchNorm2d (normalization)

Group 2: ✗ CRITICAL
  Lead Transform: conv1 (Conv2d)
  Type: transform_unit
  Impact Score: inf
  Parameters: 928 (0.2%)
  Layers (4):
    ★ conv1: Conv2d (transform)
    → bn1: BatchNorm2d (normalization)
    → relu1: ReLU (activation)
    → pool1: MaxPool2d (pooling)
```

## 🔧 How It Works

### Step 1: Identify All Transform Layers
```python
self.transform_layers = {
    'conv1': LayerInfo(...),
    'conv2a': LayerInfo(...),
    'conv2b': LayerInfo(...),
    'conv3': LayerInfo(...),
    'fc': LayerInfo(...)
}
```

### Step 2: Build Groups from Each Transform
For each transform layer:
- Start a new group with the transform as leader
- Add directly dependent non-transform layers
- Stop when reaching another transform or critical layer

### Step 3: Classify Groups
- transform_unit: Simple Conv→BN→ReLU pattern
- parallel_branch: Transform that splits to multiple paths
- complex: Other patterns

## 💡 Key Benefits

- **Clear Structure**: Every group has a well-defined leader and boundaries
- **Safe Pruning**: Removes complete computational units
- **Preserves Architecture**: Respects the fundamental building blocks
- **Easy to Understand**: Groups map directly to network blocks

## 🚀 Usage

```python
# Quick prune
pruned_model = quick_prune(model, reduction=0.3)

# Or with control
pruner = UniversalLayerPruner(model)
pruner.print_plan()
pruner.prune(max_param_reduction=0.3)
```

## Requirements

- PyTorch
- NetworkX
- NumPy

## Installation

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
