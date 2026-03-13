EoH (Evolution of Heuristics)
==============================

Background
----------

EoH (Evolution of Heuristics) is an evolutionary algorithm framework that leverages Large Language Models (LLMs) for automatic algorithm design. Proposed in the paper "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Models" (ICML 2024), EoH treats heuristic algorithms as individuals in an evolutionary population and uses LLMs to generate new candidates through evolutionary operators.

The key insight of EoH is that LLMs can understand and transform algorithmic logic, enabling the evolution of heuristics for combinatorial optimization problems. Unlike traditional genetic programming approaches that operate on low-level code mutations, EoH uses LLMs to perform high-level algorithmic transformations.

Algorithm Overview
------------------

EoH combines evolutionary search with LLM-based variation operators:

1. **Initialization (I1)**: Generate initial population by asking LLM to create new heuristics from scratch based on task description.

2. **Evolutionary Operators**:
   - **E1 (Crossover/Recombination)**: Given multiple parent algorithms, generate a completely different algorithm inspired by them.
   - **E2 (Crossover with Motivation)**: Given multiple parent algorithms, identify common backbone ideas and generate a new algorithm based on that backbone.
   - **M1 (Mutation)**: Given one parent algorithm, generate a modified version with different algorithmic structure.
   - **M2 (Parameter Tuning)**: Given one parent algorithm, modify its scoring/selection parameters.

3. **Population Management**: Maintains a population of evaluated heuristics, performs survival selection based on fitness scores.

4. **Parallel Execution**: Supports multi-threaded sampling and evaluation for efficiency.

Pseudocode
----------

.. code-block:: text

    Initialize population P = empty
    while |P| < pop_size:
        prompt = I1(task_description, template_function)
        h = LLM.generate(prompt)
        score = evaluate(h)
        if score is valid:
            P.add(h)

    while not termination_condition:
        # E1: Generate different algorithm
        parents = select(P, selection_num)
        prompt = E1(task_description, parents, template_function)
        h = LLM.generate(prompt)
        score = evaluate(h)
        P.add(h)

        # E2: Generate motivated algorithm
        if use_e2_operator:
            parents = select(P, selection_num)
            prompt = E2(task_description, parents, template_function)
            h = LLM.generate(prompt)
            score = evaluate(h)
            P.add(h)

        # M1: Mutation
        if use_m1_operator:
            parent = select(P, 1)
            prompt = M1(task_description, parent, template_function)
            h = LLM.generate(prompt)
            score = evaluate(h)
            P.add(h)

        # M2: Parameter tuning
        if use_m2_operator:
            parent = select(P, 1)
            prompt = M2(task_description, parent, template_function)
            h = LLM.generate(prompt)
            score = evaluate(h)
            P.add(h)

        P = survival(P, pop_size)

    return best_in(P)

Usage
-----

To utilize the `EoH` class, initialize it with the necessary parameters and call the `run` method to start the evolutionary process.

Constructor
-----------

.. class:: EoH

    .. rubric:: Parameters

    - **llm** (llm4ad.base.LLM): An instance of 'llm4ad.base.LLM', which provides the way to query LLM.
    - **evaluation** (llm4ad.base.Evaluation): An instance of 'llm4ad.base.Evaluation', which defines the way to calculate the score of a generated function.
    - **profiler** (llm4ad.method.eoh.EoHProfiler | None): An instance of 'llm4ad.method.eoh.EoHProfiler'. If not needed, pass 'None'.
    - **max_generations** (int | None): Terminate after evolving 'max_generations' generations or reaching 'max_sample_nums'. Pass 'None' to disable. Default: 10.
    - **max_sample_nums** (int | None): Terminate after evaluating 'max_sample_nums' functions (valid or not) or reaching 'max_generations'. Pass 'None' to disable. Default: 100.
    - **pop_size** (int | None): Population size. If 'None', EoH will auto-adjust this parameter. Default: 5.
    - **selection_num** (int): Number of selected individuals during crossover. Default: 2.
    - **use_e2_operator** (bool): Whether to use the e2 operator. Default: True.
    - **use_m1_operator** (bool): Whether to use the m1 operator. Default: True.
    - **use_m2_operator** (bool): Whether to use the m2 operator. Default: True.
    - **num_samplers** (int): Number of parallel sampler threads. Default: 1.
    - **num_evaluators** (int): Number of parallel evaluator threads. Default: 1.
    - **resume_mode** (bool): In resume_mode, 'randsample' skips evaluating 'template_program' and initialization. Default: False.
    - **debug_mode** (bool): If True, detailed information will be printed. Default: False.
    - **multi_thread_or_process_eval** (str): Use 'thread' (ThreadPoolExecutor) or 'process' (ProcessPoolExecutor) for multi-core CPU evaluation. Default: 'thread'.
    - **kwargs** (dict): Additional args passed to 'llm4ad.base.SecureEvaluator', such as 'fork_proc'.


Methods
-------

.. method:: run()

    Starts the evolutionary optimization process. If `resume_mode` is `False`, it initializes the population and then proceeds to evolve.

Private Methods
---------------

.. method:: _iteratively_use_eoh_operator()

    Performs the core evolutionary loop using E1, E2, M1, and M2 operators (when enabled) to generate new candidate functions.

.. method:: _iteratively_init_population()

    Initializes the population by repeatedly sampling and evaluating functions using the I1 operator until reaching either the population size or `_initial_sample_nums_max`.

.. method:: _sample_evaluate_register(prompt: str)

    Samples a function using the provided prompt, evaluates it, and registers it with the population and profiler. Records timing and performance metrics.

.. method:: _adjust_pop_size()

    Automatically adjusts the population size based on `max_sample_nums` if no explicit population size was provided.

.. method:: _continue_loop() -> bool

    Determines whether the evolutionary process should continue based on termination conditions.

.. method:: _multi_threaded_sampling(fn: callable, *args, **kwargs)

    Executes the given function (either initialization or evolution) using multiple threads for parallel sampling.

Complete Example
----------------

.. code-block:: python

    from llm4ad.base import LLM, Evaluation
    from llm4ad.method import EoH
    from llm4ad.tool import EoHProfiler

    # Define your task
    task_description = '''
    Design a heuristic algorithm for the Bin Packing Problem.
    The algorithm should minimize the number of bins used.
    '''

    # Define the template function
    template_program = '''
    def bin_packing(items: list, bin_capacity: int) -> list:
        """Pack items into bins.
        Args:
            items: List of item sizes.
            bin_capacity: Maximum capacity of each bin.
        Returns:
            List of bins, each bin is a list of items.
        """
        bins = []
        return bins
    '''

    # Create evaluation instance
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program,
        num_test_cases=100
    )

    # Create LLM instance
    llm = LLM(model_name='gpt-4')

    # Create profiler (optional)
    profiler = EoHProfiler(save_dir='./eoh_results')

    # Initialize EoH
    eoh = EoH(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        max_generations=20,
        max_sample_nums=200,
        pop_size=10,
        selection_num=2,
        num_samplers=2,
        num_evaluators=2
    )

    # Run optimization
    eoh.run()

    # Get results
    print(f"Total samples: {profiler.total_samples}")
    print(f"Best score: {profiler.best_score}")

References
----------

- Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Models." In Forty-first International Conference on Machine Learning (ICML). 2024.

- Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "LLM4AD: A Platform for Algorithm Design with Large Language Model." arXiv preprint arXiv:2412.17287 (2024).
