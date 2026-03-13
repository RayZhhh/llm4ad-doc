# Constructive Heuristics for Capacitated Facility Location Problem (CFLP)

## Problem

The **Capacitated Facility Location Problem (CFLP)** involves assigning customers to facilities while respecting capacity constraints.

- **Given:** Set of facilities with capacities, set of customers with demands, assignment costs
- **Objective:** Minimize total assignment cost while satisfying capacity constraints
- **Constraints:** Each customer must be assigned to exactly one facility, facility capacities must not be exceeded

## Algorithm Design Task

Constructive heuristics iteratively assign customers to facilities.

- **Inputs:** Current assignments, remaining customers, remaining capacities, customer demands, assignment costs
- **Outputs:** (customer_id, facility_id) tuple

## Evaluation

- **Dataset:** 16 instances, 50 facilities, 50 customers
- **Fitness:** Average total cost (minimized)

## Template

```python
template_program = '''
import numpy as np

def select_next_assignment(assignments: List[List[int]], remaining_customers: List[int],
                          remaining_capacities: List[int], customer_demands: List[int],
                          assignment_costs: List[List[int]]) -> Tuple[int, int]:
    """
    Constructive heuristic for the Capacitated Facility Location Problem.

    Args:
        assignments: Current assignments of customers to facilities.
        remaining_customers: List of customer indices not yet assigned.
        remaining_capacities: Remaining capacities of facilities.
        customer_demands: List of customer demands.
        assignment_costs: 2D list of assignment costs.

    Returns:
        A tuple containing: (selected_customer, selected_facility)
    """
    for customer in remaining_customers:
        min_cost = float('inf')
        selected_facility = None
        for facility in range(len(remaining_capacities)):
            if remaining_capacities[facility] >= customer_demands[customer]:
                if assignment_costs[facility][customer] < min_cost:
                    min_cost = assignment_costs[facility][customer]
                    selected_facility = facility
        if selected_facility is not None:
            return customer, selected_facility
    return None, None
'''

task_description = '''
Given facilities with capacities and customers, iteratively assign customers to facilities
while respecting capacity constraints and minimizing total costs.
Design a novel algorithm to select the next assignment in each step.
'''
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `timeout_seconds` | Maximum evaluation time | 60 |
| `n_instance` | Number of problem instances | 16 |
| `n_facilities` | Number of facilities | 50 |
| `n_customers` | Number of customers | 50 |
| `max_capacity` | Maximum facility capacity | 100 |
| `max_demand` | Maximum customer demand | 20 |

## Example Usage

```python
from llm4ad.task.optimization.cflp_construct import CFLPEvaluation

evaluator = CFLPEvaluation()

def select_next_assignment(assignments, remaining_customers, remaining_capacities,
                          customer_demands, assignment_costs):
    for customer in remaining_customers:
        min_cost = float('inf')
        selected_facility = None
        for facility in range(len(remaining_capacities)):
            if remaining_capacities[facility] >= customer_demands[customer]:
                if assignment_costs[facility][customer] < min_cost:
                    min_cost = assignment_costs[facility][customer]
                    selected_facility = facility
        if selected_facility is not None:
            return customer, selected_facility
    return None, None

result = evaluator.evaluate_program('', select_next_assignment)
```
