NSGA2
===============

Background
-----------

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is one of the most influential multi-objective evolutionary algorithms, proposed by Deb et al. in 2002. It uses non-dominated sorting to rank solutions based on their Pareto dominance and crowding distance to maintain diversity among solutions.

In the context of LLM4AD, NSGA2 adapts the NSGA-II framework for automatic algorithm design. Instead of traditional genetic operators (crossover, mutation), it employs Large Language Models as the variation operator to generate new candidate algorithms. The algorithm maintains a diverse population of heuristics across the Pareto front, optimizing multiple objectives simultaneously.

Reference paper:

- Shunyu Yao, Fei Liu, Xi Lin, Zhichao Lu, Zhenkun Wang, and Qingfu Zhang. "Multi-objective evolution of heuristic using large language model." In Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

Algorithm Overview
-----------------

The NSGA2 method in LLM4AD combines the non-dominated sorting multi-objective optimization framework with LLM-based heuristic generation. The key components are:

1. **Non-dominated Sorting**: Solutions are ranked based on Pareto dominance. The first front contains non-dominated solutions, the second front contains solutions dominated only by those in the first front, and so on.

2. **Crowding Distance**: Maintains diversity by calculating the crowding distance of each solution in the same front. Solutions in less crowded regions are preferred.

3. **Evolutionary Operators**: Five LLM-based operators are used:
   - **I1 (Initialization)**: Generate new heuristics from scratch based on task description
   - **E1 (Crossover - Different)**: Generate a new heuristic completely different from parent heuristics
   - **E2 (Crossover - Inspired)**: Generate a new heuristic inspired by the common backbone of parent heuristics
   - **M1 (Mutation)**: Modify an existing heuristic to create a new variant
   - **M2 (Parameter Tuning)**: Adjust parameters of an existing heuristic

4. **Selection**: Uses tournament selection based on rank and crowding distance. Solutions with lower rank (better Pareto front) are preferred. If ranks are equal, solutions with higher crowding distance (more diverse) are preferred.

5. **Update Strategy**: The population is updated by combining parent and offspring populations, then selecting the best solutions using non-dominated sorting and crowding distance.

Pseudocode
----------

.. code-block:: text

    Input: LLM, Evaluation, Population Size N, Max Generations G, Max Samples M
    Output: Population of evolved heuristics

    // Initialize
    Initialize population P = empty

    // Initialize population using I1 operator
    while |P| < N and samples < M:
        prompt = generate_init_prompt(task_description)
        h = LLM.generate(prompt)           // I1 operator
        score = evaluate(h)
        P.add(h)
        register_to_profiler(h)

    // Assign rank and crowding distance
    assign_rank_and_crowding(P)

    // Evolution
    generation = 0
    while continue_sampling(G, M):
        // Create offspring population Q
        Q = empty
        while |Q| < N:
            // Select parents using tournament selection
            parent1 = tournament_select(P)
            parent2 = tournament_select(P)

            // E1: Crossover (different)
            if use_e1_operator and |Q| < N:
                prompt = generate_crossover_prompt([parent1, parent2], task)
                h = LLM.generate(prompt)
                score = evaluate(h)
                Q.add(h)

            // E2: Crossover (inspired)
            if use_e2_operator and |Q| < N:
                prompt = generate_inspired_prompt([parent1, parent2], task)
                h = LLM.generate(prompt)
                score = evaluate(h)
                Q.add(h)

            // M1: Mutation
            if use_m1_operator and |Q| < N:
                prompt = generate_mutation_prompt(parent1, task)
                h = LLM.generate(prompt)
                score = evaluate(h)
                Q.add(h)

            // M2: Parameter tuning
            if use_m2_operator and |Q| < N:
                prompt = generate_param_tuning_prompt(parent1, task)
                h = LLM.generate(prompt)
                score = evaluate(h)
                Q.add(h)

        // Combine parent and offspring
        R = P + Q

        // Perform non-dominated sorting
        fronts = non_dominated_sort(R)

        // Create new population
        P_new = empty
        for each front in fronts:
            if |P_new| + |front| <= N:
                calculate_crowding_distance(front)
                P_new.add(front)
            else:
                // Fill remaining slots using crowding distance
                sort_by_crowding(front)
                P_new.add(first N - |P_new| individuals of front)
                break

        P = P_new
        assign_rank_and_crowding(P)
        generation += 1

    return P

Usage
-----

To use the `NSGA2` class, initialize it with the required parameters and call the `run` method to start the evolutionary optimization process.

Constructor
-----------

.. class:: NSGA2

    .. rubric:: Parameters

    - **llm** (LLM): An instance of `llm4ad.base.LLM` for querying the LLM.
    - **evaluation** (Evaluation): An instance of `llm4ad.base.Evaluation` to calculate scores of generated functions.
    - **profiler** (NSGA2Profiler, optional): An instance of `llm4ad.method.nsga2.NSGA2Profiler`. Pass `None` if profiling is not needed. Defaults to None.
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

    Starts the NSGA-II optimization process. If `resume_mode` is False, initializes the population before evolution.

    The run method performs the following steps:

    1. If resume_mode is False:
       - Initialize population using I1 operator (generate from task description)
       - Assign rank and crowding distance to all individuals
       - Ensure at least `selection_num` valid individuals exist
    2. Create offspring population using E1, E2, M1, M2 operators
    3. Combine parent and offspring, select best using non-dominated sorting and crowding distance
    4. Return the final population

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

    Thread worker for performing evolutionary operations. Each thread generates new solutions using LLM-based operators.

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
- **_sampler** (NSGA2Sampler): Sampler instance.
- **_evaluator** (SecureEvaluator): Evaluator instance.
- **_profiler** (NSGA2Profiler): Profiler instance.
- **_tot_sample_nums** (int): Total number of samples evaluated.
- **_evaluation_executor** (concurrent.futures.Executor): Executor for parallel evaluation.

Example
-------

.. code-block:: python

    from llm4ad.method import NSGA2
    from llm4ad.base import LLM, Evaluation
    from llm4ad.method.nsga2 import NSGA2Profiler

    # Define task
    task_description = "Design a heuristic for the vehicle routing problem that minimizes total distance and number of vehicles."

    template_program = '''
    def vehicle_routing_heuristic(depots, customers, vehicle_capacity):
        """
        A heuristic for the vehicle routing problem.

        Args:
            depots: List of depot locations
            customers: List of customer locations with demands
            vehicle_capacity: Maximum capacity of each vehicle

        Returns:
            List of routes, where each route is a list of customer indices
        """
        # TODO: Implement your heuristic here
        routes = []
        return routes
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
        objective_functions=['total_distance', 'num_vehicles']
    )

    # Initialize profiler
    profiler = Profiler(
        log_dir='./logs/nsga2',
        save_frequency=10
    )

    # Create NSGA2 optimizer
    optimizer = NSGA2(
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

    # Get Pareto front solutions
    pareto_front = result.get_pareto_front()

    # Get best solutions for each objective
    for solution in pareto_front:
        print(f"Score: {solution.score}, Algorithm: {solution.algorithm}")

Exceptions
----------

- **RuntimeError**: Raised if the initial population cannot be properly initialized (less than selection_num valid individuals).
- **ValueError**: Raised if invalid parameters are provided (e.g., invalid multi_thread_or_process_eval value).
- **AssertionError**: Raised if the multi_thread_or_process_eval parameter is neither 'thread' nor 'process'.

References
-----------

1. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.

2. Shunyu Yao, Fei Liu, Xi Lin, Zhichao Lu, Zhenkun Wang, and Qingfu Zhang. "Multi-objective evolution of heuristic using large language model." In Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

3. Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
