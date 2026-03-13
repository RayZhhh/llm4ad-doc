LHNS
==========

LHNS (LLM-based Heuristic Needle Search) is an LLM-based automatic algorithm design framework that combines metaheuristic search techniques with Large Language Models. It adapts three classical metaheuristics - Variable Neighborhood Search (VNS), Iterated Local Search (ILS), and Tabu Search (TS) - to guide the LLM in generating improved heuristics.

Background
----------

LHNS was proposed in the paper "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model" (ICML 2024). The key insight is that LLMs can be treated as powerful variation operators within established metaheuristic frameworks. By combining the reasoning capabilities of LLMs with systematic search strategies, LHNS can effectively explore the space of algorithmic heuristics.

The algorithm uses a simulated annealing-based acceptance criterion that gradually reduces exploration over time, allowing the search to converge to high-quality solutions while maintaining diversity in early stages.

Algorithm Overview
------------------

LHNS supports three search methods:

1. **Variable Neighborhood Search (VNS)**: Systematically changes neighborhood structures when no improvement is found, with an increasing exploration rate.

2. **Iterated Local Search (ILS)**: Perturbs the best found solution after reaching a local optimum, combining intensification and diversification.

3. **Tabu Search (TS)**: Uses a tabu list (elite set) to avoid revisiting recently explored solutions and promotes diversity.

All three methods share common components:
- **Elite Set**: Stores the best solutions found
- **Simulated Annealing Acceptance**: Accepts worse solutions with decreasing probability based on cooling rate
- **LLM-based Variation**: Uses LLM to generate new solutions from current/best/elite solutions

Pseudocode (VNS variant)
-----------------------

.. code-block:: text

    // Initialization
    current = LLM.generate_initial_solution()
    best = current
    cooling_rate = initial_cooling_rate
    trans_count = 0

    WHILE total_samples < max_sample_nums:
        // Generate new solution using LLM with random restart strategy
        prompt = create_rr_prompt(task, current, cooling_rate, template)
        new_solution = LLM.generate(prompt)

        // Simulated annealing acceptance
        accept = simulated_annealing_accept(current, new_solution, cooling_rate, total_samples, max_samples)

        IF accept:
            current = new_solution
            cooling_rate = initial_cooling_rate
            IF current.score > best.score:
                best = current
        ELSE:
            // Increase exploration
            IF cooling_rate < 1.0:
                cooling_rate += 0.1
            ELSE:
                cooling_rate = initial_cooling_rate

    Return best

Constructor Parameters
---------------------

.. class:: LHNS

    .. rubric:: Parameters

    - **llm** (llm4ad.base.LLM): An instance of 'llm4ad.base.LLM', which provides the way to query LLM.
    - **evaluation** (llm4ad.base.Evaluation): An instance of 'llm4ad.base.Evaluation', which defines the way to calculate the score of a generated function.
    - **profiler** (llm4ad.method.lhns.LHNSProfiler | None): An instance of 'llm4ad.method.lhns.LHNSProfiler'. If not needed, pass 'None'.
    - **max_sample_nums** (int | None): Terminate after evaluating 'max_sample_nums' functions (valid or not). Pass 'None' to disable. Default is 100.
    - **cooling_rate** (float): Initial cooling rate for simulated annealing. Default is 0.1.
    - **elite_set_size** (int): Size of the elite set for storing best solutions. Default is 5.
    - **method** (str): Search method to use: 'vns' (Variable Neighborhood Search), 'ils' (Iterated Local Search), or 'ts' (Tabu Search). Default is 'vns'.
    - **num_samplers** (int): Number of parallel samplers. Default is 1.
    - **num_evaluators** (int): Number of parallel evaluators. Default is 1.
    - **resume_mode** (bool): In resume_mode, skips evaluating 'template_program' and initialization. Default is False.
    - **debug_mode** (bool): If True, detailed information will be printed. Default is False.
    - **multi_thread_or_process_eval** (str): Use 'thread' (ThreadPoolExecutor) or 'process' (ProcessPoolExecutor) for multi-core CPU evaluation. Default is 'thread'.
    - **kwargs** (dict): Additional args passed to 'llm4ad.base.SecureEvaluator', such as 'fork_proc'.

Methods
-------

.. method:: run()

    Starts the optimization process. If `resume_mode` is False, initializes the current solution first, then runs the selected search method (VNS, ILS, or TS).

Private Methods
---------------

.. method:: method_vns()

    Implements Variable Neighborhood Search with LLM-based variation. Uses random restart strategy with cooling rate to balance exploration and exploitation.

.. method:: method_ils()

    Implements Iterated Local Search with LLM-based perturbation. After 10 unsuccessful transitions, switches to perturbing the best solution.

.. method:: method_ts()

    Implements Tabu Search with LLM-based variation. Uses elite set for tabu list and merges current solution with elite solutions after 10 unsuccessful transitions.

.. method:: simulated_annealing(next_func, cooling_rate, trans_count)

    Applies simulated annealing acceptance criterion to decide whether to accept a new solution.

.. method:: _iteratively_init()

    Initializes the algorithm by generating an initial valid solution.

.. method:: _sample_evaluate_register(prompt)

    Samples a function using the provided prompt, evaluates it, and registers it with the profiler.

.. method:: _continue_loop() -> bool

    Determines whether the search should continue based on termination conditions.

.. method:: _multi_threaded_sampling(fn, *args, **kwargs)

    Executes the given function using multiple threads for parallel sampling.

Attributes
----------

- **_template_program_str** (str): String representation of the template program to evolve
- **_task_description_str** (str): Description of the optimization task
- **_cooling_rate** (float): Initial cooling rate for simulated annealing
- **_max_sample_nums** (int): Maximum number of samples to evaluate
- **_elite_set** (EliteSet): Stores the best solutions found
- **_function_to_evolve** (LHNSFunction): The base function being evolved
- **_current_function** (LHNSFunction): Current solution in the search
- **_best_function** (LHNSFunction): Best solution found so far
- **_template_program** (LHNSProgram): Parsed template program structure
- **_sampler** (LHNSSampler): Handles LLM-based function sampling
- **_evaluator** (SecureEvaluator): Evaluates function performance
- **_profiler** (LHNSProfiler): Optional profiler for tracking search metrics
- **_tot_sample_nums** (int): Total number of samples evaluated

Example Usage
-------------

.. code-block:: python

    from llm4ad.base import Evaluation, LLM
    from llm4ad.method import LHNS
    from llm4ad.tools.llm import OpenAI

    # Define your task
    task_description = "Design a heuristic for the Vehicle Routing Problem..."

    # Define template program
    template_program = '''
    def vrp_heuristic(dist_matrix, demands, vehicle_capacity):
        """A heuristic for solving the Vehicle Routing Problem.
        Args:
            dist_matrix: Distance matrix between customers.
            demands: Demand of each customer.
            vehicle_capacity: Maximum capacity of each vehicle.
        Returns:
            List of routes, one per vehicle.
        """
        n = len(demands)
        unvisited = set(range(1, n))
        routes = []

        while unvisited:
            route = [0]
            capacity = vehicle_capacity

            while True:
                nearest = None
                min_dist = float('inf')
                for customer in unvisited:
                    if demands[customer] <= capacity:
                        dist = dist_matrix[route[-1]][customer]
                        if dist < min_dist:
                            min_dist = dist
                            nearest = customer

                if nearest is None:
                    break

                route.append(nearest)
                capacity -= demands[nearest]
                unvisited.remove(nearest)

            route.append(0)
            routes.append(route)

        return routes
    '''

    # Create evaluation
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program
    )

    # Create LLM
    llm = OpenAI(model='gpt-4')

    # Create and run LHNS with VNS method
    lhns = LHNS(
        llm=llm,
        evaluation=evaluation,
        max_sample_nums=100,
        cooling_rate=0.1,
        elite_set_size=5,
        method='vns'
    )
    lhns.run()

    # Alternative: Run with Tabu Search
    # lhns = LHNS(llm=llm, evaluation=evaluation, method='ts', ...)

References
----------

- Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model." In Forty-first International Conference on Machine Learning (ICML), 2024.

- Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "LLM4AD: A Platform for Algorithm Design with Large Language Model." arXiv preprint arXiv:2412.17287 (2024).
