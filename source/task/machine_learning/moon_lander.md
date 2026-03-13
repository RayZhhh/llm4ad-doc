# Control Strategy for Lunar Lander

## Problem

The **Lunar Lander** problem involves landing a spacecraft safely on a target area.

- **Given:** Lunar lander with position, velocity, angle, and contact sensors
- **Objective:** Land safely at the target location with minimal fuel
- **Constraints:** Limited thrust, must maintain safe landing parameters

## Algorithm Design Task

Design a control strategy that selects appropriate action based on lander state.

- **Inputs:** x/y coordinates, velocities, angle, angular velocity, leg contacts, last action
- **Outputs:** Action (0=do nothing, 1=left engine, 2=main engine, 3=right engine)

## Evaluation

- **Environment:** Gym LunarLander-v2
- **Fitness:** Average reward over 5 episodes

## Template

```python
template_program = '''
import numpy as np

def choose_action(xc: float, yc: float, xv: float, yv: float, a: float,
                 av: float, lc: float, rc: float, last_action: int) -> int:
    """
    Args:
        xc: x coordinate, between [-1, 1]
        yc: y coordinate, between [-1, 1]
        xv: x velocity
        yv: y velocity
        a: angle
        av: angular velocity
        lc: 1 if first leg has contact, else 0
        rc: 1 if second leg has contact, else 0
        last_action: last move

    Return:
         Action: 0=do nothing, 1=left, 2=up, 3=right
    """
    action = np.random.randint(4)
    return action
'''

task_description = ("Implement a novel heuristic strategy that guides the lander to achieve "
                    "a safe landing at the center of the target area.")
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_steps` | Maximum steps per episode | 500 |
| `timeout_seconds` | Maximum evaluation time | 20 |

## Example Usage

```python
from llm4ad.task.machine_learning.moon_lander import MoonLanderEvaluation

evaluator = MoonLanderEvaluation()

def choose_action(xc, yc, xv, yv, a, av, lc, rc, last_action):
    if yc > 0.5 and yv < -0.1:
        return 2  # Main engine
    if abs(a) > 0.2:
        return 1 if a > 0 else 3  # Side thrusters
    return 0

result = evaluator.evaluate_program('', choose_action)
```
