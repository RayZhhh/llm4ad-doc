Introduction
============

Introduction
------------
The following figure (indicating a single iteration for updating and optimizing algorithms) demonstrates a typical pipline for the LLM-based altomatic algorithm design methods.
The solutions of LLM4AD are represented as executable functions.

.. figure:: ../assets/figs/llm-eps.png
    :alt: llm-eps
    :align: center
    :width: 100%

The main logic of this canonical paradigm can be grouped into four parts:

1. Maintaining a solution set, which contains multiple candidate solutions.

2. Select multiple candidate solution (may in favor of that with higher performance scores) and create prompt using prompt engineering.

3. Query LLM and process the response content (extract the code of algorithm from the response content).

4. Evaluate (with secure evaluation technique) the algorithm and update the solution set.

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


