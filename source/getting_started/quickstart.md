Quickstart
==========

This guide will help you get started with LLM4AD in 5 minutes.

Installation
------------

First, install LLM4AD:

.. code-block:: bash

    pip install llm4ad

Or install from source:

.. code-block:: bash

    git clone https://github.com/Optima-CityU/LLM4AD.git
    cd LLM4AD
    pip install -r requirements.txt

.. important::
    Python 3.9 or higher is required for the ``ast.unparse()`` function.

Basic Concepts
--------------

LLM4AD has three core components:

1. **LLM**: The language model interface for generating algorithms.
   Subclass ``llm4ad.base.LLM`` and implement ``draw_sample()`` to connect to your LLM API.

2. **Evaluation**: The task definition that provides templates and evaluates generated algorithms.
   Available tasks include Online Bin Packing, TSP, VRP, and more.

3. **Method**: The algorithm design method that uses the LLM to discover better algorithms.
   Methods include RandSample, EoH, FunSearch, HillClimb, and others.

Complete Example
---------------

Here's a complete example using the **Online Bin Packing** task with a fake LLM (for debugging without an API):

.. code-block:: python

    from __future__ import annotations

    import pickle
    import random
    from typing import Any
    import sys

    sys.path.append('.')

    from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
    from llm4ad.base import LLM
    from llm4ad.method.randsample import RandSample
    from llm4ad.tools.profiler import ProfilerBase


    class FakeLLM(LLM):
        """Fake LLM that randomly selects functions from a database.
        Use this for debugging without an actual LLM API.
        """

        def __init__(self):
            super().__init__()
            with open('./example/llms/online_bin_packing_fake/_data/rand_function.pkl', 'rb') as f:
                self._functions = pickle.load(f)

        def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
            return random.choice(self._functions)


    if __name__ == '__main__':
        # Define the LLM (using FakeLLM for demonstration)
        llm = FakeLLM()

        # Define the evaluation task (Online Bin Packing)
        task = OBPEvaluation()

        # Define the method (Random Sampling)
        method = RandSample(
            llm=llm,
            profiler=ProfilerBase(log_dir='logs/quickstart', log_style='simple'),
            evaluation=task,
            max_sample_nums=10,      # Maximum number of algorithm samples
            num_samplers=1,          # Number of parallel samplers
            num_evaluators=1,        # Number of parallel evaluators
        )

        # Run the algorithm design process
        method.run()

        print("Quickstart completed! Check the logs/quickstart directory for results.")

Running the Example
-------------------

Run the example script:

.. code-block:: bash

    python quickstart_example.py

You should see output indicating the algorithm design process is running. The profiler will log results to the ``logs/quickstart`` directory.

Using a Real LLM
----------------

To use a real LLM instead of the fake one, replace ``FakeLLM`` with an actual LLM implementation:

.. code-block:: python

    from llm4ad.tool import OnlineAPI

    llm = OnlineAPI(
        api_endpoint='your-api-endpoint',  # e.g., "api.bltcy.top" (no https://)
        api_key='your-api-key',            # e.g., "sk-..."
    )

Using Different Methods
-----------------------

LLM4AD supports multiple methods. Here are some examples:

**EoH (Evolution of Heuristics)**:

.. code-block:: python

    from llm4ad.method.eoh import EoH

    method = EoH(
        llm=llm,
        evaluation=task,
        max_generations=10,
        max_sample_nums=100,
        pop_size=5
    )

**FunSearch**:

.. code-block:: python

    from llm4ad.method.funsearch import FunSearch

    method = FunSearch(
        llm=llm,
        evaluation=task,
        max_sample_nums=100,
        num_samplers=4
    )

**HillClimb**:

.. code-block:: python

    from llm4ad.method.hillclimb import HillClimb

    method = HillClimb(
        llm=llm,
        evaluation=task,
        max_sample_nums=50
    )

Using Built-in Tasks
--------------------

LLM4AD provides pre-built tasks for common optimization problems:

**TSP (Traveling Salesman Problem)**:

.. code-block:: python

    from llm4ad.task.optimization.tsp_construct import TSPEvaluation

    task = TSPEvaluation()

**CVRP (Capacitated Vehicle Routing Problem)**:

.. code-block:: python

    from llm4ad.task.optimization.cvrp_construct import CVRPEvaluation

    task = CVRPEvaluation()

**Circle Packing**:

.. code-block:: python

    from llm4ad.task.optimization.circle_packing import CirclePackingEvaluation

    task = CirclePackingEvaluation()

Next Steps
----------

- Read the :doc:`installation` guide for detailed installation instructions
- Explore :doc:`examples` to see more example scripts
- Check the API documentation for available methods and tasks
- Visit the GitHub repository for more examples: https://github.com/Optima-CityU/LLM4AD
