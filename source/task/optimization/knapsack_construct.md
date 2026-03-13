# Constructive Heuristics for Knapsack Problem

## Problem

The **Knapsack Problem** is a classic combinatorial optimization problem where items with weights and values must be selected to maximize total value without exceeding capacity.

- **Given:** A set of items, each with a weight and value, and a knapsack with limited capacity
- **Objective:** Maximize the total value of selected items
- **Constraints:** Total weight cannot exceed knapsack capacity

## Algorithm Design Task

Constructive heuristics iteratively select items to add to the knapsack. The task is to design the heuristic for selecting the next item in each iteration.

- **Inputs:** Remaining capacity, remaining items (weight, value, index)
- **Outputs:** Selected item (weight, value, index) or None

## Evaluation

- **Dataset:** 32 instances, 50 items, capacity = 100
- **Fitness:** Average total value (maximized)

## Template

```python
template_program = '''
import numpy as np

def select_next_item(remaining_capacity: int, remaining_items: List[Tuple[int, int, int]]) -> Tuple[int, int, int] | None:
    """
    Select the item with the highest value-to-weight ratio that fits in the remaining capacity.

    Args:
        remaining_capacity: The remaining capacity of the knapsack.
        remaining_items: List of tuples containing (weight, value, index) of remaining items.

    Returns:
        The selected item as a tuple (weight, value, index), or None if no item fits.
    """
    best_item = None
    best_ratio = -1

    for item in remaining_items:
        weight, value, index = item
        if weight <= remaining_capacity:
            ratio = value / weight
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item

    return best_item
'''

task_description = '''
Given a set of items with weights and values, the goal is to select a subset of items
that maximizes the total value while not exceeding the knapsack's capacity.
Help me design a novel algorithm to select the next item in each step.
'''
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `timeout_seconds` | Maximum evaluation time | 20 |
| `n_instance` | Number of problem instances | 32 |
| `n_items` | Number of items | 50 |
| `knapsack_capacity` | Knapsack capacity | 100 |

## Example Usage

```python
from llm4ad.task.optimization.knapsack_construct import KnapsackEvaluation

evaluator = KnapsackEvaluation()

def select_next_item(remaining_capacity, remaining_items):
    # Greedy by value-to-weight ratio
    best_item = None
    best_ratio = -1
    for item in remaining_items:
        weight, value, index = item
        if weight <= remaining_capacity:
            ratio = value / weight
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item
    return best_item

result = evaluator.evaluate_program('', select_next_item)
```
