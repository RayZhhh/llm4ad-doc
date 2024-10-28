Welcome to LLM4AD Docs!
=======================

.. figure:: ./assets/figs/framework.pdf
    :alt: llm-eps
    :align: center
    :width: 100%


📦About this repo
-----------------
This repository implements state-of-the-art large language model based evlutionary program search (LLM-EPS) methods. We have implemented four LLM based EPS methods as follows.

+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| Methods                                               | Paper title                                                                                                              |
+=======================================================+==========================================================================================================================+
| FunSearch                                             | Mathematical Discoveries from Program Search with Large Language Models (Nature 2023)                                    |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| EoH*                                                  | Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model (ICML 2024)             |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| (1+1)-EPS* (HillClimbing)                             | Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models (PPSN 2024) |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| RandomSampling                                        | Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models (PPSN 2024) |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+

.. note::
    "*": The implementation has some minor differences from the original method (demonstrated in their original paper), considering generality and multithreading acceleration.

.. tip::
    We also provide LLM-free examples to help understanding/building the pipeline of these methods!


💡Features
----------
- Unified interfaces for multiple methods.
- Evaluation acceleration: multiprocessing evaluation, add numba wrapper for algorithms.
- Secure evaluation: main process protection, timeout interruption.
- Log: Wandb and Tensorboard support.
- Resume run supported.
- We provide easy-to-use pakage for manipulating the code and secure evaluation that help future development.


🧭Navigation
------------

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    getting_started/installation
    getting_started/examples
    getting_started/online_demo
    getting_started/gui


.. toctree::
    :maxdepth: 1
    :caption: Developer Documentation

    dev/base/base_index
    dev/base_tutorial/base_tutorial_index
    dev/platform_structure
    dev/run_new_task


.. toctree::
    :maxdepth: 1
    :caption: Method

    method/eoh
    method/funsearch
    method/hillclimb
    method/randsample


.. toctree::
    :maxdepth: 1
    :caption: Task

    task/obp
    task/cvrp

