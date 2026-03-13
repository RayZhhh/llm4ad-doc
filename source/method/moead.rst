MOEAD
===============

Background
-----------

MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) is a classical multi-objective optimization algorithm that decomposes a multi-objective problem into a set of single-objective subproblems using scalarizing functions. Originally proposed by Zhang and Li in 2007, MOEA/D has become one of the most influential multi-objective evolutionary algorithms.

In the context of LLM4AD, MOEAD adapts the MOEA/D framework for automatic algorithm design. Instead of using traditional genetic operators, it employs Large Language Models as the variation operator to generate new candidate algorithms. Each subproblem in the decomposition is associated with a weight vector representing different trade-offs between objectives, and the LLM generates new heuristics based on the solutions stored in neighboring subproblems.

Reference paper:

- Shunyu Yao, Fei Liu, Xi Lin, Zhichao Lu, Zhenkun Wang, and Qingfu Zhang. "Multi-objective evolution of heuristic using large language model." In Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

Algorithm Overview
-----------------

The MOEAD method in LLM4AD combines the decomposition-based multi-objective optimization framework with LLM-based heuristic generation. The key components are:

1. **Decomposition**: The population is organized according to weight vectors that represent different Pareto-optimal regions. Each weight vector defines a subproblem with a scalarizing function (Tchebycheff approach).

2. **Evolutionary Operators**: Four LLM-based operators are used:
   - **I1 (Initialization)**: Generate new heuristics from scratch based on task description
   - **E1 (Crossover - Different)**: Generate a new heuristic completely different from parent heuristics
   - **E2 (Crossover - Inspired)**: Generate a new heuristic inspired by the common backbone of parent heuristics
   - **M1 (Mutation)**: Modify an existing heuristic to create a new variant
   - **M2 (Parameter Tuning)**: Adjust parameters of an existing heuristic

3. **Selection**: Uses weight vector-based selection - individuals are selected based on their compatibility with the subproblem's weight vector using the Tchebycheff scalarizing function.

4. **Update Strategy**: When a new solution is generated, it updates its neighboring subproblems if it improves their objective values.

Pseudocode
----------

.. code-block:: text

    Input: LLM, Evaluation, Population Size N, Max Generations G, Max Samples M
    Output: Population of evolved heuristics

    // Initialize
    Initialize weight vectors W = {w1, w2, ..., wN}
    Initialize population P = empty

    // Initialize population using I1 operator
    while |P| < N and samples < M:
        prompt = generate_init_prompt(task_description)
        h = LLM.generate(prompt)           // I1 operator
        score = evaluate(h)
        P.add(h)
        register_to_profiler(h)

    // Evolution
    while continue_sampling(G, M):
        for each weight vector wi in W:
            // Select parents based on weight vector
            parents = select_parents(P, wi, selection_num)

            // E1: Crossover (different)
            if use_e1_operator:
                prompt = generate_crossover_prompt(parents, task)
                h = LLM.generate(prompt)
                score = evaluate(h)
                P.update_with_new_solution(h, wi)

            // E2: Crossover (inspired)
            if use_e2_operator:
                prompt = generate_inspired_prompt(parents, task)
                h = LLM.generate(prompt)
                score = evaluate(h)
                P.update_with_new_solution(h, wi)

            // M1: Mutation
            if use_m1_operator:
                parent = select_one(P, wi)
                prompt = generate_mutation_prompt(parent, task)
                h = LLM.generate(prompt)
                score = evaluate(h)
                P.update_with_new_solution(h, wi)

            // M2: Parameter tuning
            if use_m2_operator:
                parent = select_one(P, wi)
                prompt = generate_param_tuning_prompt(parent, task)
                h = LLM.generate(prompt)
                score = evaluate(h)
                P.update_with_new_solution(h, wi)

    return P

Usage
-----

To use the `MOEAD` class, initialize it with the required parameters and call the `run` method to start the evolutionary optimization process.

Constructor
-----------

.. class:: MOEAD

    .. rubric:: Parameters

    - **llm** (LLM): An instance of `llm4ad.base.LLM` for querying the LLM.
    - **evaluation** (Evaluation): An instance of `llm4ad.base.Evaluation` to calculate scores of generated functions.
    - **profiler** (MOEADProfiler, optional): An instance of `llm4ad.method.moead.MOEADProfiler`. Pass `None` if profiling is not needed. Defaults to None.
    - **max_generations** (int | None, optional): Maximum number of generations to evolve. Defaults to 10.
    - **max_sample_nums** (int | None, optional): Maximum number of functions to evaluate. Defaults to 100.
    - **pop_size** (int, optional): Population size. Defaults to 20.
    - **selection_num** (int, optional): Number of selected individuals for crossover. Defaults to 5.
    - **use_e2_operator** (bool, optional): Whether to use the E2 evolutionary operator (crossover inspired). Defaults to True.
    - **use_m1_operator** (bool, optional): Whether to use the M1 evolutionary operator (mutation). Defaults to True.
    - **use_m2_operator** (bool, optional): Whether to use the M2 evolutionary operator (parameter tuning). Defaults to True.
    - **num_samplers** (int, optional): Number of sampler threads. Defaults to 1.
    - **num_evaluators** (int, optional): Number of evaluator threads. Defaults to 1.
    - **num_objs** (int, optional): Number of objectives. Defaults to 2.
    - **resume_mode** (bool, optional): If True, skips initial evaluation and initialization. Defaults to False.
    - **initial_sample_num** (int | None, optional): Initial sample count. Defaults to None.
    - **debug_mode** (bool, optional): If True, prints detailed information. Defaults to False.
    - **multi_thread_or_process_eval** (str, optional): Use 'thread' or 'process' for evaluation. Defaults to 'thread'.
    - **kwargs**: Additional arguments passed to `llm4ad.base.SecureEvaluator`.

Methods
-------

.. method:: run()

    Starts the MOEA/D optimization process. If `resume_mode` is False, initializes the population before evolution.

    The run method performs the following steps:

    1. If resume_mode is False:
       - Initialize population using I1 operator (generate from task description)
       - Ensure at least `selection_num` valid individuals exist
    2. Execute evolutionary process using E1, E2, M1, M2 operators
    3. Return the final population

Private Methods
---------------

.. method:: _sample_evaluate_register(prompt)

    Samples a function using the given prompt, evaluates it, and registers it to the population.

    Parameters:
        - prompt (str): The prompt to send to the LLM

.. method:: _continue_sample() -> bool

    Checks if sampling should continue based on generation and sample count limits.

    Returns:
        - bool: True if sampling should continue, False otherwise

.. method:: _thread_do_evolutionary_operator()

    Thread worker for performing evolutionary operations. Each thread processes operators based on weight vectors.

.. method:: _thread_init_population()

    Thread worker for initializing the population using the I1 operator.

.. method:: _init_population()

    Initializes the population using multiple threads.

.. method:: _do_sample()

    Performs the evolutionary sampling process using multiple threads.

Attributes
----------

- **_template_program_str** (str): The template program string.
- **_task_description_str** (str): The task description string.
- **_num_objs** (int): Number of objectives.
- **_max_generations** (int | None): Maximum generations.
- **_max_sample_nums** (int | None): Maximum samples.
- **_pop_size** (int): Population size.
- **_selection_num** (int): Selection number for crossover.
- **_use_e2_operator** (bool): E2 operator flag.
- **_use_m1_operator** (bool): M1 operator flag.
- **_use_m2_operator** (bool): M2 operator flag.
- **_num_samplers** (int): Number of samplers.
- **_num_evaluators** (int): Number of evaluators.
- **_resume_mode** (bool): Resume mode flag.
- **_debug_mode** (bool): Debug mode flag.
- **_function_to_evolve** (Function): Function being evolved.
- **_function_to_evolve_name** (str): Name of function being evolved.
- **_template_program** (Program): Template program instance.
- **_population** (Population): Population instance.
- **_sampler** (MOEADSampler): Sampler instance.
- **_evaluator** (SecureEvaluator): Evaluator instance.
- **_profiler** (MOEADProfiler): Profiler instance.
- **_tot_sample_nums** (int): Total number of samples evaluated.
- **_evaluation_executor** (concurrent.futures.Executor): Executor for parallel evaluation.

Example
-------

.. code-block:: python

    from llm4ad.method import MOEAD
    from llm4ad.base import LLM, Evaluation
    from llm4ad.method.moead import MOEADProfiler

    # Define task
    task_description = "Design a heuristic for the bin packing problem that minimizes the number of bins used."

    template_program = '''
    def bin_packing_heuristic(items, bin_capacity):
        """
        A heuristic for the bin packing problem.

        Args:
            items: List of item weights
            bin_capacity: Maximum capacity of each bin

        Returns:
            List of bins, where each bin is a list of item indices
        """
        # TODO: Implement your heuristic here
        bins = []
        return bins
    '''

    # Initialize LLM (e.g., OpenAI GPT-4)
    llm = LLM(
        model_name='gpt-4',
        api_key='your-api-key'
    )

    # Define evaluation
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program,
        num_test_cases=100,
        objective_functions=['efficiency', 'simplicity']
    )

    # Initialize profiler
    profiler = Profiler(
        log_dir='./logs/moead',
        save_frequency=10
    )

    # Create MOEAD optimizer
    optimizer = MOEAD(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        max_generations=20,
        max_sample_nums=200,
        pop_size=20,
        num_samplers=2,
        num_evaluators=4
    )

    # Run optimization
    result = optimizer.run()

    # Get best solutions
    best_solutions = result.get_pareto_front()

Exceptions
----------

- **RuntimeError**: Raised if the initial population cannot be properly initialized (less than selection_num valid individuals).
- **ValueError**: Raised if invalid parameters are provided (e.g., invalid multi_thread_or_process_eval value).
- **AssertionError**: Raised if the multi_thread_or_process_eval parameter is neither 'thread' nor 'process'.

References
-----------

1. Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on evolutionary computation, 11(6), 712-731.

2. Shunyu Yao, Fei Liu, Xi Lin, Zhichao Lu, Zhenkun Wang, and Qingfu Zhang. "Multi-objective evolution of heuristic using large language model." In Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

3. Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
