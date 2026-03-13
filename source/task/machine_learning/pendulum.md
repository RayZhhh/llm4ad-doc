# Control Strategy for Pendulum Swing-up Problem

## Problem

The **Pendulum Swing-up Problem** is a classic control problem in reinforcement learning.

- **Given:** A pendulum that starts in a downward position
- **Objective:** Swing the pendulum to upright position and stabilize it
- **Constraints:** Torque limited to [-2.0, 2.0]

## Algorithm Design Task

Design a control strategy that selects appropriate torque based on pendulum state.

- **Inputs:** cos(theta), sin(theta), angular velocity, last action
- **Outputs:** Torque to apply (float in [-2.0, 2.0])

## Evaluation

- **Environment:** Gym Pendulum-v1
- **Fitness:** Average reward over 5 episodes

## Template

```python
template_program = '''
import numpy as np

def choose_action(x: float, y: float, av: float, last_action: float) -> float:
    """
    Args:
        x: cos(theta), between [-1, 1]
        y: sin(theta), between [-1, 1]
        av: angular velocity of the pendulum, between [-8.0, 8.0]
        last_action: the last torque applied, between [-2.0, 2.0]

    Return:
         A float representing the torque to be applied.
         The value should be in the range of [-2.0, 2.0].
    """
    action = np.random.uniform(-2.0, 2.0)
    return action
'''

task_description = ("Implement a novel control strategy for the inverted pendulum swing-up problem. "
                    "The goal is to apply an appropriate torque at each step to swing the pendulum "
                    "into an upright position and stabilize it.")
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_steps` | Maximum steps per episode | 500 |
| `timeout_seconds` | Maximum evaluation time | 20 |

## Example Usage

```python
from llm4ad.task.machine_learning.pendulum import PendulumEvaluation

evaluator = PendulumEvaluation()

def choose_action(x, y, av, last_action):
    # Simple swing-up strategy
    if y < 0:  # Below horizontal
        return -2.0 if x > 0 else 2.0
    else:
        return -0.5 * av - 2.0 * x

result = evaluator.evaluate_program('', choose_action)
```
