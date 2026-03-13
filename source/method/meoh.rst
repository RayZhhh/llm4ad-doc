MEoH (Multi-objective Evolution of Heuristics)
===============================================

Background
----------

MEoH (Multi-objective Evolution of Heuristics) extends the EoH framework to handle multi-objective optimization problems. Proposed in the paper "Multi-objective evolution of heuristic using large language model" (AAAI 2025), MEoH maintains a population of heuristics that are optimized simultaneously across multiple objectives, such as solution quality and computational efficiency.

The key insight of MEoH is that heuristic algorithms often need to balance multiple competing objectives (e.g., speed vs. accuracy, exploration vs. exploitation). By using multi-objective evolutionary optimization with LLMs, MEoH can discover a diverse set of Pareto-optimal heuristics that represent different trade-offs between objectives.

Algorithm Overview
------------------

MEoH combines multi-objective evolutionary algorithms with LLM-based variation operators:

1. **Initialization (I1)**: Generate initial population by asking LLM to create new heuristics from scratch.

2. **Multi-objective Evaluation**: Each heuristic is evaluated on multiple objectives (e.g., primary score, execution time, complexity).

3. **Evolutionary Operators**:
   - **E1 (Crossover)**: Given multiple parent algorithms, generate a completely different algorithm.
   - **E2 (Crossover with Motivation)**: Identify common backbone ideas and generate a new algorithm.
   - **M1 (Mutation)**: Generate a modified version with different algorithmic structure.
   - **M2 (Parameter Tuning)**: Modify scoring/selection parameters.

4. **Pareto-based Selection**: Uses Pareto dominance to rank individuals and maintain diverse solutions.

5. **Population Management**: Maintains population of non-dominated solutions across objectives.

Pseudocode
----------

.. code-block:: text

    Initialize population P = empty
    while |P| < pop_size:
        prompt = I1(task_description, template_function)
        h = LLM.generate(prompt)
        scores = evaluate_multi_objective(h)  # Returns vector of scores
        if all scores are valid:
            P.add(h, scores)

    while not termination_condition:
        # E1: Generate different algorithm
        parents = select(P, selection_num)
        prompt = E1(task_description, parents, template_function)
        h = LLM.generate(prompt)
        scores = evaluate_multi_objective(h)
        P.add(h, scores)

        # E2: Generate motivated algorithm
        if use_e2_operator:
            parents = select(P, selection_num)
            prompt = E2(task_description, parents, template_function)
            h = LLM.generate(prompt)
            scores = evaluate_multi_objective(h)
            P.add(h, scores)

        # M1: Mutation
        if use_m1_operator:
            parent = select(P, 1)
            prompt = M1(task_description, parent, template_function)
            h = LLM.generate(prompt)
            scores = evaluate_multi_objective(h)
            P.add(h, scores)

        # M2: Parameter tuning
        if use_m2_operator:
            parent = select(P, 1)
            prompt = M2(task_description, parent, template_function)
            h = LLM.generate(prompt)
            scores = evaluate_multi_objective(h)
            P.add(h, scores)

        P = survival_multi_objective(P, pop_size)

    return Pareto_front(P)

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
    - **selection_num** (int): Number of selected individuals for crossover. Defaults to 2.
    - **use_e2_operator** (bool): Whether to use the E2 (crossover) operator. Defaults to True.
    - **use_m1_operator** (bool): Whether to use the M1 (mutation) operator. Defaults to True.
    - **use_m2_operator** (bool): Whether to use the M2 (mutation) operator. Defaults to True.
    - **num_samplers** (int): Number of sampler threads. Defaults to 1.
    - **num_evaluators** (int): Number of evaluator threads. Defaults to 1.
    - **num_objs** (int): Number of optimization objectives. Defaults to 2.
    - **resume_mode** (bool): If `True`, skips initial evaluation and resumes from a previous state. Defaults to False.
    - **initial_sample_num** (int | None): Initial sample count. Defaults to None.
    - **debug_mode** (bool): If `True`, prints detailed debug information. Defaults to False.
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

Complete Example
----------------

.. code-block:: python

    from llm4ad.base import LLM, Evaluation
    from llm4ad.method import MEoH
    from llm4ad.tool import MEoHProfiler

    # Define your task with multiple objectives
    task_description = '''
    Design a heuristic algorithm for the Vehicle Routing Problem.
    Objective 1: Minimize total distance traveled.
    Objective 2: Minimize number of vehicles used.
    Objective 3: Minimize computational time.
    '''

    # Define the template function
    template_program = '''
    def vehicle_routing(customers: list, depot: tuple, vehicle_capacity: int) -> dict:
        """Plan vehicle routes.
        Args:
            customers: List of customer locations and demands.
            depot: Depot location.
            vehicle_capacity: Maximum capacity of each vehicle.
        Returns:
            Dictionary with routes and total distance.
        """
        routes = []
        return {'routes': routes, 'total_distance': 0}
    '''

    # Create evaluation instance with multi-objective metrics
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program,
        num_test_cases=100
    )

    # Create LLM instance
    llm = LLM(model_name='gpt-4')

    # Create profiler (optional)
    profiler = MEoHProfiler(save_dir='./meoh_results')

    # Initialize MEoH
    meoh = MEoH(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        max_generations=20,
        max_sample_nums=200,
        pop_size=20,
        selection_num=2,
        num_objs=2,  # Primary score + execution time
        num_samplers=2,
        num_evaluators=2
    )

    # Run optimization
    meoh.run()

    # Get results
    print(f"Total samples: {profiler.total_samples}")
    print(f"Pareto front size: {len(profiler.pareto_front)}")

References
----------

- Shunyu Yao, Fei Liu, Xi Lin, Zhichao Lu, Zhenkun Wang, and Qingfu Zhang. "Multi-objective evolution of heuristic using large language model." In Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

- Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "LLM4AD: A Platform for Algorithm Design with Large Language Model." arXiv preprint arXiv:2412.17287 (2024).
