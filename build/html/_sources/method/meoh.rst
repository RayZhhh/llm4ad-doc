MEoH
==========

The `MEoH` class implements a multi-objective evolutionary algorithm to optimize a given program using evolutionary operators (crossover and mutation) with LLM-based sampling and evaluation.

Usage
-----

To use the `MEoH` class, initialize it with the required parameters and call the run method to start the evolutionary optimization process.

Constructor
-----------

.. class:: MEoH

    .. rubric:: Parameters  

    - **llm** (LLM): An instance of `llm4ad.base.LLM`, providing the interface to query the LLM.  
    - **evaluation** (Evaluation): An instance of `llm4ad.base.Evaluation`, defining how to score generated functions.
    - **profiler** (MEoHProfiler, optional): An instance of `llm4ad.method.meoh.MEoHProfiler`. Pass `None` if profiling is not needed.  
    - **max_generations** (int | None): Terminate after evolving `max_generations` generations or reaching `max_sample_nums`. Defaults to 10.  
    - **max_sample_nums** (int | None): Terminate after evaluating `max_sample_nums` functions (valid or invalid) or reaching `max_generations`. Defaults to 100.  
    - **pop_size** (int): Population size. Defaults to 20.  
    - **selection_num** (int): Number of selected individuals for crossover. Defaults to 5.  
    - **use_e2_operator** (bool): Whether to use the E2 (crossover) operator. Defaults to `True`.  
    - **use_m1_operator** (bool): Whether to use the M1 (mutation) operator. Defaults to `True`.  
    - **use_m2_operator** (bool): Whether to use the M2 (mutation) operator. Defaults to `True`.  
    - **num_samplers** (int): Number of sampler threads. Defaults to 1.  
    - **num_evaluators** (int): Number of evaluator threads. Defaults to 1.  
    - **num_objs** (int): Number of optimization objectives. Defaults to 2.  
    - **resume_mode** (bool): If `True`, skips initial evaluation and resumes from a previous state. Defaults to `False`.  
    - **initial_sample_num** (int | None): Initial sample count. Defaults to `None`.  
    - **debug_mode** (bool): If `True`, prints detailed debug information. Defaults to `False`.  
    - **multi_thread_or_process_eval** (str): Use 'thread' or 'process' for parallel evaluation. Defaults to 'thread'.  
    - **kwargs**: Additional arguments passed to `llm4ad.base.SecureEvaluator`.  
    

Methods
-------

.. method:: run()

Starts the evolutionary optimization process. If `resume_mode` is `False`, it initializes the population and then evolves it using evolutionary operators.  

Private Methods
---------------

.. method:: _sample_evaluate_register(prompt)

Samples a function using the given prompt, evaluates it, and registers it in the population.  
.. method:: _continue_sample() -> bool

Checks whether sampling should continue based on `max_generations` and `max_sample_nums`.  
.. method:: _thread_do_evolutionary_operator()

Worker thread function that applies evolutionary operators (E1, E2, M1, M2) to generate new candidates.  
.. method:: _thread_init_population()

Worker thread function for initializing the population using the I1 operator.  
.. method:: _init_population()

Initializes the population by sampling and evaluating initial candidates.  
.. method:: _do_sample()

Executes the evolutionary sampling process using multiple threads.  

Attributes
----------

- **_template_program_str** (str): The string representation of the template program.
- **_task_description_str** (str): The task description for LLM prompting.
- **_num_objs** (int): Number of optimization objectives.
- **_max_generations** (int | None): Maximum generations allowed.
- **_max_sample_nums** (int | None): Maximum samples allowed.
- **_pop_size** (int): Population size.
- **_selection_num** (int): Number of selected individuals for crossover.
- **_use_e2_operator** (bool): Whether E2 operator is enabled.
- **_use_m1_operator** (bool): Whether