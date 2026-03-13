# Control Strategy for Continuous Mountain Car

## Problem

The **Continuous Mountain Car** problem involves controlling a car to reach a target on a hill.

- **Given:** Car position and velocity on an uneven terrain
- **Objective:** Reach the target position at the top of the hill
- **Constraints:** Force limited to [-1.0, 1.0]

## Algorithm Design Task

Design a control strategy that selects appropriate force based on car state.

- **Inputs:** Position, velocity, last action
- **Outputs:** Force to apply (float in [-1.0, 1.0])

## Evaluation

- **Environment:** Gym MountainCarContinuous-v0
- **Fitness:** Episode outcome (succeeded/failed)

## Template

```python
template_program = '''
import numpy as np

def choose_action(pos: float, v: float, last_action: float) -> float:
    """
    Args:
        pos: Car's position, between [-1.2, 0.6]
        v: Car's velocity, between [-0.07, 0.07]
        last_action: Car's last force, between [-1, 1]

    Return:
         A float representing the force to be applied.
         The value should be in the range of [-1.0, 1.0].
    """
    return np.random.uniform(-1.0, 1.0)
'''

task_description = ("Implement a function that designs a novel strategy function that "
                    "guides the car along an uneven road towards a target.")
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_steps` | Maximum steps per episode | 500 |
| `timeout_seconds` | Maximum evaluation time | 20 |

## Example Usage

```python
from llm4ad.task.machine_learning.car_mountain_continue import CarMountainCEvaluation

evaluator = CarMountainCEvaluation()

def choose_action(pos, v, last_action):
    if pos < 0:
        return -1.0  # Go left to build momentum
    else:
        return 1.0  # Go right towards goal

result = evaluator.evaluate_program('', choose_action)
```
