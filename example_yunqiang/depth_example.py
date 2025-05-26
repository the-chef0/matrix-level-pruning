"""
Universal Layer Grouping Strategy for Pruning
Transform-layer-centric grouping: All groups start from computational layers
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Dict, Tuple, Set, Optional, Any, Union
import re
from collections import OrderedDict, defaultdict, deque
import networkx as nx
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LayerInfo:
    """Information about a layer for pruning decisions."""
    name: str
    module: nn.Module
    type: str
    params: int
    is_critical: bool = False
    dependencies: Set[str] = field(default_factory=set)
    group_id: Optional[int] = None
    importance_score: float = 0.0


@dataclass 
class PruningGroup:
    """A group of layers that should be pruned together."""
    id: int
    layers: List[LayerInfo]
    type: str  # 'transform_unit', 'parallel_branch', 'complex'
    lead_transform: LayerInfo  # The transform layer that leads this group
    can_remove: bool = True
    removal_impact: float = 0.0
    dependencies: Set[int] = field(default_factory=set)


class TransformCentricPruningGrouper:
    """
    Transform-centric grouping strategy that ensures all groups start from computational layers.
    
    Core Principles:
    1. Every group MUST start with a transform layer (Conv, Linear, etc.)
    2. Non-transform layers can only be grouped with their preceding transform layer
    3. Transform layers define the boundaries of pruning groups
    4. Safety and correctness are paramount
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = OrderedDict()
        self.transform_layers = OrderedDict()  # Only transform layers
        self.graph = nx.DiGraph()
        self.groups = OrderedDict()
        self.critical_layers = set()
        
        # Layer type categories
        self.TRANSFORM_LAYERS = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear,
                                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        self.NORM_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                           nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                           nn.InstanceNorm2d, nn.InstanceNorm3d)
        self.ACTIVATION_LAYERS = (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU,
                                 nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid, nn.Softmax,
                                 nn.ReLU6, nn.Hardswish, nn.Hardsigmoid)
        self.POOLING_LAYERS = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                              nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                              nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d,
                              nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool1d,
                              nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)
        self.DROPOUT_LAYERS = (nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                              nn.AlphaDropout)
        self.RESHAPE_LAYERS = (nn.Flatten, nn.Unflatten)
        
        # Initialize analysis
        self._analyze_model()
    
    def _analyze_model(self):
        """Comprehensive model analysis with transform-centric approach."""
        # Step 1: Collect all layers
        self._collect_layers()
        
        # Step 2: Build dependency graph
        self._build_dependency_graph()
        
        # Step 3: Identify critical layers
        self._identify_critical_layers()
        
        # Step 4: Calculate importance scores
        self._calculate_importance_scores()
        
        # Step 5: Form transform-centric groups
        self._form_transform_centric_groups()
    
    def _collect_layers(self):
        """Collect all layers with special attention to transform layers."""
        for name, module in self.model.named_modules():
            if name and not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                layer_type = self._get_layer_type(module)
                params = sum(p.numel() for p in module.parameters())
                
                layer_info = LayerInfo(
                    name=name,
                    module=module,
                    type=layer_type,
                    params=params
                )
                
                self.layers[name] = layer_info
                
                # Keep separate track of transform layers
                if isinstance(module, self.TRANSFORM_LAYERS):
                    self.transform_layers[name] = layer_info
    
    def _get_layer_type(self, module: nn.Module) -> str:
        """Categorize layer type."""
        if isinstance(module, self.TRANSFORM_LAYERS):
            return 'transform'
        elif isinstance(module, self.NORM_LAYERS):
            return 'normalization'
        elif isinstance(module, self.ACTIVATION_LAYERS):
            return 'activation'
        elif isinstance(module, self.POOLING_LAYERS):
            return 'pooling'
        elif isinstance(module, self.DROPOUT_LAYERS):
            return 'dropout'
        elif isinstance(module, self.RESHAPE_LAYERS):
            return 'reshape'
        else:
            # Check for merge/split operations
            module_name = module.__class__.__name__.lower()
            if any(op in module_name for op in ['concat', 'cat', 'merge', 'add', 'sum']):
                return 'merge'
            elif any(op in module_name for op in ['split', 'chunk', 'slice']):
                return 'split'
            else:
                return 'other'
    
    def _build_dependency_graph(self):
        """Build dependency graph using multiple strategies."""
        # Try different strategies
        if not self._try_fx_trace():
            if not self._try_hook_analysis():
                self._heuristic_graph_building()
    
    def _try_fx_trace(self) -> bool:
        """Try to build graph using torch.fx."""
        try:
            sample_input = self._create_sample_input()
            traced = fx.symbolic_trace(self.model)
            
            fx_to_layer = {}
            for node in traced.graph.nodes:
                if node.op == 'call_module':
                    fx_to_layer[node.name] = node.target
            
            for node in traced.graph.nodes:
                if node.op == 'call_module' and node.target in self.layers:
                    self.graph.add_node(node.target)
                    
                    for input_node in node.args:
                        if hasattr(input_node, 'name') and input_node.name in fx_to_layer:
                            if fx_to_layer[input_node.name] in self.layers:
                                self.graph.add_edge(fx_to_layer[input_node.name], node.target)
            
            return self.graph.number_of_nodes() > 0
        except:
            return False
    
    def _try_hook_analysis(self) -> bool:
        """Try to analyze using forward hooks."""
        try:
            execution_order = []
            handles = []
            
            def create_hook(name):
                def hook(module, input, output):
                    execution_order.append(name)
                return hook
            
            for name, module in self.model.named_modules():
                if name and name in self.layers:
                    handle = module.register_forward_hook(create_hook(name))
                    handles.append(handle)
            
            sample_input = self._create_sample_input()
            with torch.no_grad():
                _ = self.model(sample_input)
            
            for handle in handles:
                handle.remove()
            
            # Build graph from execution order
            for i in range(len(execution_order) - 1):
                self.graph.add_edge(execution_order[i], execution_order[i + 1])
            
            return len(execution_order) > 0
        except:
            return False
    
    def _heuristic_graph_building(self):
        """Build graph using heuristics focused on transform layer connections."""
        layer_list = list(self.layers.keys())
        
        # First pass: Connect transform layers to their dependent layers
        for i, name in enumerate(layer_list):
            layer = self.layers[name]
            
            if layer.type == 'transform':
                # Look ahead for dependent non-transform layers
                j = i + 1
                while j < len(layer_list):
                    next_name = layer_list[j]
                    next_layer = self.layers[next_name]
                    
                    # Stop at the next transform layer
                    if next_layer.type == 'transform':
                        # Check if they're directly connected
                        if self._are_directly_connected(name, next_name):
                            self.graph.add_edge(name, next_name)
                        break
                    
                    # Connect non-transform layers if they follow this transform
                    if self._should_connect_to_transform(name, next_name, layer, next_layer):
                        self.graph.add_edge(name, next_name)
                        
                        # Chain normalization -> activation -> dropout
                        if j + 1 < len(layer_list):
                            if next_layer.type == 'normalization':
                                third_name = layer_list[j + 1]
                                third_layer = self.layers[third_name]
                                if third_layer.type in ['activation', 'dropout']:
                                    self.graph.add_edge(next_name, third_name)
                    
                    j += 1
        
        # Second pass: Handle parallel branches (multiple transforms from same source)
        for name in self.transform_layers:
            predecessors = list(self.graph.predecessors(name))
            if len(predecessors) == 1:
                pred_name = predecessors[0]
                # Find other transforms with same predecessor
                for other_name in self.transform_layers:
                    if other_name != name:
                        other_preds = list(self.graph.predecessors(other_name))
                        if other_preds == predecessors:
                            # These are parallel branches
                            # Mark the predecessor as a split point
                            if pred_name in self.layers:
                                self.layers[pred_name].is_critical = True
    
    def _should_connect_to_transform(self, transform_name: str, candidate_name: str,
                                    transform_layer: LayerInfo, candidate_layer: LayerInfo) -> bool:
        """Determine if a non-transform layer should be connected to a transform layer."""
        # Check naming patterns
        if self._share_base_name(transform_name, candidate_name):
            return True
        
        # Check sequential numbering
        if self._are_sequentially_numbered(transform_name, candidate_name):
            return True
        
        # Check layer type patterns (conv -> bn -> relu is common)
        if candidate_layer.type in ['normalization', 'activation', 'dropout']:
            # These often follow transform layers
            return True
        
        return False
    
    def _are_directly_connected(self, name1: str, name2: str) -> bool:
        """Check if two transform layers are likely directly connected."""
        # Sequential numbering in same module
        if self._are_sequentially_numbered(name1, name2):
            return True
        
        # Parent-child relationship
        if name2.startswith(name1 + '.') or name1.startswith(name2 + '.'):
            return True
        
        return False
    
    def _are_sequentially_numbered(self, name1: str, name2: str) -> bool:
        """Check if layers have sequential numbering."""
        match1 = re.search(r'(\d+)', name1)
        match2 = re.search(r'(\d+)', name2)
        
        if match1 and match2:
            num1 = int(match1.group(1))
            num2 = int(match2.group(1))
            
            # Remove numbers to get base names
            base1 = re.sub(r'\d+', '', name1)
            base2 = re.sub(r'\d+', '', name2)
            
            return base1 == base2 and abs(num2 - num1) == 1
        
        return False
    
    def _share_base_name(self, name1: str, name2: str) -> bool:
        """Check if two layers share a base name."""
        # Remove numbers and common suffixes
        base1 = re.sub(r'[_\.]?\d+', '', name1)
        base2 = re.sub(r'[_\.]?\d+', '', name2)
        
        for suffix in ['conv', 'bn', 'relu', 'act', 'norm', 'linear', 'fc', 'dropout']:
            base1 = re.sub(f'[_\.]?{suffix}$', '', base1)
            base2 = re.sub(f'[_\.]?{suffix}$', '', base2)
        
        return base1 == base2 and base1 != ''
    
    def _identify_critical_layers(self):
        """Identify layers that are critical for network function."""
        # 1. First and last transform layers
        transform_names = list(self.transform_layers.keys())
        if transform_names:
            # First transform layer
            self.transform_layers[transform_names[0]].is_critical = True
            self.critical_layers.add(transform_names[0])
            
            # Last transform layer
            self.transform_layers[transform_names[-1]].is_critical = True
            self.critical_layers.add(transform_names[-1])
        
        # 2. Merge and split operations
        for name, layer in self.layers.items():
            if layer.type in ['merge', 'split']:
                layer.is_critical = True
                self.critical_layers.add(name)
        
        # 3. Transform layers with multiple outputs (split points)
        for name in self.transform_layers:
            out_degree = self.graph.out_degree(name)
            successors = list(self.graph.successors(name))
            
            # Count transform successors
            transform_successors = sum(1 for s in successors 
                                     if s in self.transform_layers)
            
            if transform_successors > 1:
                # This transform feeds multiple other transforms
                self.layers[name].is_critical = True
                self.critical_layers.add(name)
        
        # 4. Transform layers that are merge points
        for name in self.transform_layers:
            in_degree = self.graph.in_degree(name)
            predecessors = list(self.graph.predecessors(name))
            
            # Count transform predecessors
            transform_predecessors = sum(1 for p in predecessors 
                                       if p in self.transform_layers)
            
            if transform_predecessors > 1:
                # Multiple transforms merge into this one
                self.layers[name].is_critical = True
                self.critical_layers.add(name)
    
    def _calculate_importance_scores(self):
        """Calculate importance score for each layer."""
        total_params = sum(layer.params for layer in self.layers.values())
        
        for name, layer in self.layers.items():
            # Base score from parameter count
            param_score = layer.params / total_params if total_params > 0 else 0
            
            # Connectivity score (for transform layers)
            connectivity_score = 0
            if layer.type == 'transform':
                in_degree = self.graph.in_degree(name)
                out_degree = self.graph.out_degree(name)
                connectivity_score = (in_degree + out_degree) / len(self.layers)
            
            # Type-based score
            type_scores = {
                'transform': 1.0,      # Most important
                'merge': 2.0,          # Critical
                'split': 2.0,          # Critical
                'normalization': 0.6,  # Important but not standalone
                'activation': 0.4,     # Usually paired with transform
                'pooling': 0.7,        # Can be important
                'reshape': 0.8,        # Important for dimensions
                'dropout': 0.2,        # Least important
                'other': 0.3
            }
            type_score = type_scores.get(layer.type, 0.3)
            
            # Critical layer boost
            critical_score = 2.0 if layer.is_critical else 1.0
            
            # Combined score (weighted differently for transform vs non-transform)
            if layer.type == 'transform':
                layer.importance_score = (
                    param_score * 0.5 +
                    connectivity_score * 0.3 +
                    type_score * 0.1 +
                    critical_score * 0.1
                )
            else:
                layer.importance_score = (
                    param_score * 0.2 +
                    type_score * 0.6 +
                    critical_score * 0.2
                )
    
    def _form_transform_centric_groups(self):
        """Form groups starting from transform layers."""
        group_id = 0
        assigned_layers = set()
        
        # Process each transform layer
        for transform_name, transform_layer in self.transform_layers.items():
            if transform_name in assigned_layers:
                continue
            
            # Form a group starting from this transform
            group = self._form_group_from_transform(transform_name, assigned_layers)
            
            if group:
                # Determine if the group can be removed
                can_remove = not any(layer.is_critical for layer in group)
                
                pruning_group = PruningGroup(
                    id=group_id,
                    layers=group,
                    type=self._classify_group_type(group),
                    lead_transform=transform_layer,
                    can_remove=can_remove
                )
                
                self._evaluate_group_removal_impact(pruning_group)
                self.groups[group_id] = pruning_group
                group_id += 1
        
        # Sort groups by removal priority
        self._prioritize_groups()
    
    def _form_group_from_transform(self, transform_name: str, 
                                   assigned: Set[str]) -> List[LayerInfo]:
        """Form a group starting from a transform layer."""
        if transform_name in assigned:
            return []
        
        group = []
        transform_layer = self.transform_layers[transform_name]
        
        # Always start with the transform layer
        group.append(transform_layer)
        assigned.add(transform_name)
        
        # Find all non-transform layers that depend on this transform
        successors = list(self.graph.successors(transform_name))
        
        for successor_name in successors:
            if successor_name in assigned:
                continue
            
            successor = self.layers.get(successor_name)
            if not successor:
                continue
            
            # Only include non-transform layers in this group
            if successor.type != 'transform':
                # Check if this layer is exclusively dependent on our transform
                predecessors = list(self.graph.predecessors(successor_name))
                
                if len(predecessors) == 1 and predecessors[0] == transform_name:
                    group.append(successor)
                    assigned.add(successor_name)
                    
                    # Continue following the chain
                    self._follow_non_transform_chain(successor_name, group, assigned)
        
        return group
    
    def _follow_non_transform_chain(self, start_name: str, group: List[LayerInfo], 
                                   assigned: Set[str]):
        """Follow chain of non-transform layers."""
        current = start_name
        
        while True:
            successors = list(self.graph.successors(current))
            if not successors:
                break
            
            # Find non-transform successors
            found_next = False
            for successor_name in successors:
                if successor_name in assigned:
                    continue
                
                successor = self.layers.get(successor_name)
                if successor and successor.type != 'transform':
                    # Check if it only depends on layers in our group
                    predecessors = list(self.graph.predecessors(successor_name))
                    if all(p in assigned for p in predecessors):
                        group.append(successor)
                        assigned.add(successor_name)
                        current = successor_name
                        found_next = True
                        break
            
            if not found_next:
                break
    
    def _classify_group_type(self, group: List[LayerInfo]) -> str:
        """Classify the type of a transform-centric group."""
        # Check if it's a simple transform unit (transform + norm + activation)
        if len(group) <= 4:
            types = [layer.type for layer in group]
            if 'transform' in types and any(t in types for t in ['normalization', 'activation']):
                return 'transform_unit'
        
        # Check if it's part of a parallel branch
        transform = group[0]  # First layer is always transform
        if self.graph.out_degree(transform.name) > 1:
            return 'parallel_branch'
        
        return 'complex'
    
    def _evaluate_group_removal_impact(self, group: PruningGroup):
        """Evaluate the impact of removing a group."""
        # Parameter reduction
        total_params = sum(p.numel() for p in self.model.parameters())
        group_params = sum(layer.params for layer in group.layers)
        param_impact = group_params / total_params if total_params > 0 else 0
        
        # Importance of the lead transform
        transform_importance = group.lead_transform.importance_score
        
        # Average importance of all layers
        avg_importance = np.mean([layer.importance_score for layer in group.layers])
        
        # Connectivity impact - how many other transforms depend on this
        lead_name = group.lead_transform.name
        dependent_transforms = 0
        
        for successor in self.graph.successors(lead_name):
            if successor in self.transform_layers:
                dependent_transforms += 1
        
        connectivity_impact = dependent_transforms / len(self.transform_layers) if self.transform_layers else 0
        
        # Combined impact (lower is better for removal)
        group.removal_impact = (
            param_impact * 0.4 +
            transform_importance * 0.3 +
            avg_importance * 0.2 +
            connectivity_impact * 0.1
        )
        
        # Critical groups have infinite impact
        if not group.can_remove:
            group.removal_impact = float('inf')
    
    def _prioritize_groups(self):
        """Sort groups by removal priority."""
        sorted_groups = sorted(
            self.groups.items(),
            key=lambda x: (not x[1].can_remove, x[1].removal_impact)
        )
        
        self.groups = OrderedDict(sorted_groups)
    
    def _create_sample_input(self) -> torch.Tensor:
        """Create sample input for model tracing."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                return torch.randn(1, module.in_channels, 224, 224)
            elif isinstance(module, nn.Conv1d):
                return torch.randn(1, module.in_channels, 100)
            elif isinstance(module, nn.Linear):
                return torch.randn(1, module.in_features)
        
        return torch.randn(1, 3, 224, 224)
    
    def get_pruning_groups(self) -> List[PruningGroup]:
        """Get all pruning groups sorted by removal priority."""
        return list(self.groups.values())
    
    def get_safe_removal_groups(self, max_param_reduction: float = 0.3) -> List[PruningGroup]:
        """Get groups that can be safely removed within parameter budget."""
        safe_groups = []
        total_params = sum(p.numel() for p in self.model.parameters())
        removed_params = 0
        
        for group in self.groups.values():
            if not group.can_remove:
                continue
            
            group_params = sum(layer.params for layer in group.layers)
            if removed_params + group_params <= total_params * max_param_reduction:
                safe_groups.append(group)
                removed_params += group_params
        
        return safe_groups
    
    def print_pruning_plan(self, max_param_reduction: float = 0.3):
        """Print a detailed pruning plan."""
        print("TRANSFORM-CENTRIC PRUNING ANALYSIS")
        print("=" * 80)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Total layers: {len(self.layers)}")
        print(f"Transform layers: {len(self.transform_layers)}")
        print(f"Critical layers: {len(self.critical_layers)}")
        
        print("\n\nPRUNING GROUPS (each starting from a transform layer):")
        print("-" * 80)
        
        safe_groups = self.get_safe_removal_groups(max_param_reduction)
        total_removable_params = 0
        
        for i, group in enumerate(self.groups.values()):
            group_params = sum(layer.params for layer in group.layers)
            param_reduction = (group_params / total_params * 100) if total_params > 0 else 0
            
            status = "✓ SAFE TO REMOVE" if group in safe_groups else (
                "✗ CRITICAL" if not group.can_remove else "○ OPTIONAL"
            )
            
            print(f"\nGroup {group.id}: {status}")
            print(f"  Lead Transform: {group.lead_transform.name} ({group.lead_transform.module.__class__.__name__})")
            print(f"  Type: {group.type}")
            print(f"  Impact Score: {group.removal_impact:.3f}")
            print(f"  Parameters: {group_params:,} ({param_reduction:.1f}%)")
            print(f"  Layers ({len(group.layers)}):")
            
            for layer in group.layers:
                prefix = "  → " if layer.name != group.lead_transform.name else "  ★ "
                print(f"  {prefix}{layer.name}: {layer.module.__class__.__name__} ({layer.type})")
            
            if group in safe_groups:
                total_removable_params += group_params
        
        print("\n\nPRUNING SUMMARY:")
        print("-" * 80)
        print(f"Safe groups to remove: {len(safe_groups)}")
        print(f"Total parameters to remove: {total_removable_params:,}")
        print(f"Parameter reduction: {total_removable_params/total_params*100:.1f}%")
        print(f"Remaining parameters: {total_params - total_removable_params:,}")


class UniversalLayerPruner:
    """Simple interface for pruning layers using transform-centric grouping."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.grouper = TransformCentricPruningGrouper(model)
        self.removed_groups = []
    
    def prune(self, max_param_reduction: float = 0.3) -> nn.Module:
        """Automatically prune the model up to the specified parameter reduction."""
        safe_groups = self.grouper.get_safe_removal_groups(max_param_reduction)
        
        for group in safe_groups:
            self._remove_group(group)
            self.removed_groups.append(group)
        
        return self.model
    
    def prune_group(self, group_id: int) -> nn.Module:
        """Remove a specific group by ID."""
        if group_id in self.grouper.groups:
            group = self.grouper.groups[group_id]
            if group.can_remove:
                self._remove_group(group)
                self.removed_groups.append(group)
            else:
                print(f"Warning: Group {group_id} is critical and cannot be removed.")
        
        return self.model
    
    def _remove_group(self, group: PruningGroup):
        """Remove all layers in a group."""
        for layer in group.layers:
            self._remove_layer(layer.name)
    
    def _remove_layer(self, layer_name: str):
        """Replace a layer with Identity."""
        parts = layer_name.split('.')
        parent = self.model
        
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        if parts[-1].isdigit():
            parent[int(parts[-1])] = nn.Identity()
        else:
            setattr(parent, parts[-1], nn.Identity())
    
    def print_plan(self):
        """Print the pruning plan."""
        self.grouper.print_pruning_plan()


# Example usage
def example_transform_centric_pruning():
    """Demonstrate transform-centric pruning."""
    
    print("EXAMPLE: Transform-Centric Pruning")
    print("=" * 80)
    
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Block 1
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            
            # Block 2 - parallel branches
            self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2a = nn.BatchNorm2d(64)
            self.relu2a = nn.ReLU()
            
            self.conv2b = nn.Conv2d(32, 64, 5, padding=2)
            self.bn2b = nn.BatchNorm2d(64)
            
            # Block 3 - merge and continue
            self.conv3 = nn.Conv2d(128, 128, 1)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu3 = nn.ReLU()
            
            # Output
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            # Block 1
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            
            # Parallel branches
            xa = self.conv2a(x)
            xa = self.bn2a(xa)
            xa = self.relu2a(xa)
            
            xb = self.conv2b(x)
            xb = self.bn2b(xb)
            
            # Merge
            x = torch.cat([xa, xb], dim=1)
            
            # Continue
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            
            # Output
            x = self.avgpool(x)
            x = x.flatten(1)
            x = self.fc(x)
            
            return x
    
    model = ExampleModel()
    pruner = UniversalLayerPruner(model)
    
    print("MODEL STRUCTURE:")
    print("-" * 40)
    for name, module in model.named_modules():
        if name:
            layer_type = 'TRANSFORM' if isinstance(module, pruner.grouper.TRANSFORM_LAYERS) else 'auxiliary'
            print(f"{name}: {module.__class__.__name__} ({layer_type})")
    
    print("\n")
    pruner.print_plan()
    
    # Show specific grouping
    print("\n\nDETAILED GROUP ANALYSIS:")
    print("-" * 80)
    for group in pruner.grouper.groups.values():
        print(f"\nGroup {group.id}:")
        print(f"  Starts from: {group.lead_transform.name} (TRANSFORM)")
        print(f"  Includes:")
        for layer in group.layers:
            if layer.name == group.lead_transform.name:
                print(f"    ★ {layer.name} (TRANSFORM - group leader)")
            else:
                print(f"    → {layer.name} (auxiliary - depends on leader)")
    
    # prune the model for one step 
    pruned_model = pruner.prune(max_param_reduction = 0.3)
    print("PRUNED MODEL STRUCTURE:")
    print("-" * 40)
    for name, module in pruned_model.named_modules():
        if name:
            layer_type = 'TRANSFORM' if isinstance(module, pruner.grouper.TRANSFORM_LAYERS) else 'auxiliary'
            print(f"{name}: {module.__class__.__name__} ({layer_type})")


def quick_prune(model: nn.Module, reduction: float = 0.3) -> nn.Module:
    """Quick function to prune any model with transform-centric approach."""
    pruner = UniversalLayerPruner(model)
    return pruner.prune(reduction)


if __name__ == "__main__":
    example_transform_centric_pruning()