Welcome to LLM4AD Docs!
=======================

**Large language model for algorithm design (LLM4AD) platform** has established an efficient, large language model-based framework for algorithm design,
aimed at assisting researchers and related users in this field to conduct experimental exploration and industrial applications more quickly and conveniently.


.. figure:: ./assets/figs/llm-eps.png
    :alt: llm-eps
    :align: center
    :width: 100%


🔥News
--------
- **Nov 5th, 2024:** LLM4AD 1.0.0 is released.


🚀Comming soon
---------------
- Other programming languages support.


🌟Features
------------
- LLM4AD unifies interfaces for multiple methods.
- Evaluation acceleration: multiprocessing evaluation, add numba wrapper for algorithms.
- Secure evaluation: main process protection, timeout interruption.
- Log and profiling: Wandb and Tensorboard support.
- Resume run supported.
- LLM4AD provides package for code modification and secure evaluation for future development.
- LLM4AD is with GUI support.


📦Supported methods
-----------------------
LLM4AD implements state-of-the-art large language model based evlutionary program search (LLM-EPS) methods. We have implemented four LLM based EPS methods as follows.

+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| Methods                                               | Paper title                                                                                                              |
+=======================================================+==========================================================================================================================+
| FunSearch                                             | Mathematical Discoveries from Program Search with Large Language Models (Nature 2023)                                    |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| EoH💡                                                 | Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model (ICML 2024)             |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| (1+1)-EPS💡(HillClimbing)                             | Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models (PPSN 2024) |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| RandomSampling                                        | Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models (PPSN 2024) |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+

.. note::
    💡: The implementation has some minor differences from the original method (demonstrated in their original paper), considering generality and multithreading acceleration.

.. tip::
    We also provide LLM-free examples to help understanding/building the pipeline of these methods!


🚌Supported tasks
----------------------
LLM4AD provides various example tasks including machine learning, optimization, and science discovery.

+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| Domain                                                | Tasks                                                                                                                    |
+=======================================================+==========================================================================================================================+
| Machine learning                                      | Acrobot, Cart Pole, Mountain Var                                                                                         |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| Optimization                                          | CVRP, OVRP, TSP, VRPTW                                                                                                   |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| Science discovery                                     | Bacteria Grow, Ordinary Differential Equation, Oscillator 1, Oscillator 2, Stress Strain                                 |
+-------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+


📝About (do we add this?)
--------------------------
- This framework is powered by anyoptimization, a Python research community. It is developed and maintained by Julian Blank who is affiliated to the Computational Optimization and Innovation Laboratory (COIN) supervised by Kalyanmoy Deb at the Michigan State University in East Lansing, Michigan, USA.

- We have developed the framework for research purposes and hope to contribute to the research area by delivering tools for solving and analyzing multi-objective problems. Each algorithm is developed as close as possible to the proposed version to the best of our knowledge. NSGA-II and NSGA-III have been developed collaboratively with one of the authors and, therefore, we recommend using them for official benchmarks.

- If you intend to use our framework for any profit-making purposes, please contact us. Also, be aware that even state-of-the-art algorithms are just the starting point for many optimization problems. The full potential of genetic algorithms requires customization and the incorporation of domain knowledge. We have experience for more than 20 years in the optimization field and are eager to tackle challenging problems. Let us know if you are interested in working with experienced collaborators in optimization. Please keep in mind that only through such projects can we keep developing and improving our framework and making sure it meets the industry’s current needs.

- Moreover, any kind of contribution is more than welcome:
    1. Give us a star on GitHub. This makes not only our framework but, in general, multi-objective optimization more accessible by being listed with a higher rank regarding specific keywords.
    2. To offer more and more new algorithms and features, we are more than happy if somebody wants to contribute by developing code. You can see it as a win-win situation because your development will be linked to your publication(s), which can significantly increase your work awareness. Please note that we aim to keep a high level of code quality, and some refactoring might be suggested.
    3. You like our framework, and you would like to use it for profit-making purposes? We are always searching for industrial collaborations because they help direct research to meet the industry’s needs. Our laboratory solving practical problems have a high priority for every student and can help you benefit from the research experience we have gained over the last years.

- If you find a bug or you have any kind of concern regarding the correctness, please use our Issue Tracker Nobody is perfect Moreover, only if we are aware of the issues we can start to investigate them.



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

    dev/platform_structure
    dev/base/index
    dev/base_tutorial/index
    dev/run_new_task
    dev/llm


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

    task/machine_learning/index
    task/optimization/index
    task/science_discovery/index

