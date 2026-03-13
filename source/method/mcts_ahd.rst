MCTS-AHD
==========

MCTS-AHD (Monte Carlo Tree Search for Automatic Heuristic Design) is an LLM-based algorithm design framework that combines Monte Carlo Tree Search (MCTS) with Large Language Models. It treats the space of algorithmic heuristics as a search tree and uses MCTS to systematically explore and exploit promising variations, balancing exploration of new solution patterns with exploitation of high-quality solutions.

Background
----------

MCTS-AHD was proposed to address the challenge of comprehensively exploring the space of algorithmic heuristics using LLMs. Traditional LLM-based evolutionary algorithms often rely on random variation operators, which can lead to redundant exploration or missing promising regions of the search space.

The key innovation of MCTS-AHD is the application of the UCT (Upper Confidence Bound for Trees) formula to guide the search, similar to its successful use in game playing. This allows the algorithm to:

- Explore diverse solution patterns through selective operators
- Exploit high-quality solutions found so far
- Balance between exploration and exploitation dynamically

The algorithm defines several evolutionary operators (E1, E2, M1, M2, S1) as actions in the MCTS framework, using the UCT formula to select which operator to apply at each step.

The UCT formula used is:

.. math::

    UCT(node) = \frac{Q - Q_{min}}{Q_{max} - Q_{min}} + \lambda \sqrt{\frac{\ln(parent.visits + 1)}{node.visits}}

Where:
- Q is the average reward of the node
- Q_min and Q_max are the minimum and maximum Q values in the tree
- lambda is the exploration constant (controlled by lambda_0 and remaining budget)
- alpha controls the tree depth expansion threshold

Algorithm Overview
------------------

MCTS-AHD operates through the following stages:

1. **Initialization**: Generate an initial solution and populate the tree root.

2. **Tree Expansion**: Expand the root node by generating multiple child solutions using the E1 operator.

3. **Tree Search**: For each iteration:
   - Traverse the tree using UCT to select promising nodes
   - Apply an evolutionary operator (E1, E2, M1, M2, or S1) based on weighted selection
   - Evaluate the new solution
   - Update tree statistics through backpropagation

4. **Population Management**: Maintain a population of best solutions and perform survival selection.

5. **Termination**: Continue until the sample budget is exhausted.

The operators work as follows:
- **E1 (Evolution from Root)**: Generate new solutions from the root using population information
- **E2 (Evolution from Population)**: Crossover between current node and a randomly selected population member
- **M1 (Mutation 1)**: Mutate the current solution with random modifications
- **M2 (Mutation 2)**: Another mutation strategy for diversity
- **S1 (Selection)**: Select and combine solutions from the path

Pseudocode
----------

.. code-block:: text

    // Initialization
    Initialize population with random solutions
    Initialize MCTS tree with root
    Expand root with initial population

    WHILE total_samples < max_sample_nums:
        // Tree traversal using UCT
        current = root
        WHILE current has children:
            uct_scores = [UCT(child) for child in current.children]
            current = child with max(uct_scores)

            // Tree expansion when visits exceed children count
            IF current.visits^alpha > len(current.children):
                IF current is root:
                    expand(current, E1)
                ELSE:
                    expand(current, E2)

        // Operator selection and application
        FOR each operator in [E1, E2, M1, M2, S1]:
            apply_operator_weight times:
                new_solution = operator(current, population)
                Evaluate new_solution
                Add to tree and population
                Backpropagate(new_solution)

        // Population survival
        population.survival()

    Return best solution in population

Constructor Parameters
---------------------

.. class:: MCTS_AHD

    .. rubric:: Parameters

    - **llm** (llm4ad.base.LLM): An instance of 'llm4ad.base.LLM', which provides the way to query LLM.
    - **evaluation** (llm4ad.base.Evaluation): An instance of 'llm4ad.base.Evaluation', which defines the way to calculate the score of a generated function.
    - **profiler** (llm4ad.method.mcts_ahd.MAProfiler | None): An instance of 'llm4ad.method.mcts_ahd.MAProfiler', 'llm4ad.method.mcts_ahd.MATensorboardProfiler', or 'llm4ad.method.mcts_ahd.MAWandbProfiler'. If not needed, pass 'None'.
    - **max_sample_nums** (int | None): Terminate after evaluating 'max_sample_nums' functions (valid or not). Pass 'None' to disable. Default is 100.
    - **init_size** (int | float | None): Initial population size. If 'None', MCTS-AHD will auto-adjust this parameter. Default is 4.
    - **pop_size** (int | None): Population size. If 'None', MCTS-AHD will auto-adjust this parameter. Default is 10.
    - **selection_num** (int): Number of selected individuals for crossover. Default is 2.
    - **num_samplers** (int): Number of parallel LLM samplers. Default is 1.
    - **num_evaluators** (int): Number of parallel evaluators using ThreadPoolExecutor or ProcessPoolExecutor. Default is 1.
    - **alpha** (float): Parameter for the UCT formula balancing exploration and exploitation. Default is 0.5.
    - **lambda_0** (float): Parameter for the UCT formula controlling exploration constant. Default is 0.1.
    - **resume_mode** (bool): In resume_mode, skips evaluating 'template_program' and initialization. Default is False.
    - **debug_mode** (bool): If True, detailed information will be printed. Default is False.
    - **multi_thread_or_process_eval** (str): Use 'thread' (ThreadPoolExecutor) or 'process' (ProcessPoolExecutor) for multi-core CPU evaluation. Default is 'thread'.
    - **kwargs** (dict): Additional args passed to 'llm4ad.base.SecureEvaluator', such as 'fork_proc'.

Methods
-------

.. method:: run()

    Starts the optimization process. Initializes the MCTS tree, populates it with initial solutions, and runs the tree search until the sample budget is exhausted.

Private Methods
---------------

.. method:: expand(mcts, node_set, cur_node, option)

    Expands the MCTS tree by applying the specified operator (E1, E2, M1, M2, or S1) to generate new solutions.

.. method:: _iteratively_init_population_root()

    Initializes the population by generating solutions from the root using E1 operator.

.. method:: _init_one_solution()

    Initializes by generating a single valid solution.

.. method:: _sample_evaluate_register(prompt, func_only)

    Samples a function using the provided prompt, evaluates it, and optionally registers it to the population and profiler.

.. method:: population_management_s1(pop_input, size)

    Manages population by keeping the top-k unique individuals.

.. method:: check_duplicate(population, code)

    Checks if a solution already exists in the population.

.. method:: _continue_loop() -> bool

    Determines whether the search should continue based on termination conditions.

.. method:: _multi_threaded_sampling(fn, *args, **kwargs)

    Executes the given function using multiple threads for parallel sampling.

Attributes
----------

- **_template_program_str** (str): String representation of the template program to evolve
- **_task_description_str** (str): Description of the optimization task
- **_max_sample_nums** (int): Maximum number of samples to evaluate
- **_init_pop_size** (int): Initial population size
- **_pop_size** (int): Main population size
- **_selection_num** (int): Number of individuals for crossover
- **alpha** (float): UCT exploration parameter
- **lambda_0** (float): UCT exploration parameter
- **_function_to_evolve** (Function): The base function being evolved
- **_function_to_evolve_name** (str): Name of the function being evolved
- **_template_program** (Program): Parsed template program structure
- **_population** (Population): Manages current population of candidate functions
- **_sampler** (MASampler): Handles LLM-based function sampling
- **_evaluator** (SecureEvaluator): Evaluates function performance
- **_profiler** (MAProfiler): Optional profiler for tracking search metrics
- **_tot_sample_nums** (int): Total number of samples evaluated

Example Usage
-------------

Basic Example
^^^^^^^^^^^^^

.. code-block:: python

    from llm4ad.base import Evaluation, LLM
    from llm4ad.method import MCTS_AHD
    from llm4ad.tools.llm import OpenAI

    # Define your task
    task_description = "Design a heuristic for the Knapsack Problem..."

    # Define template program
    template_program = '''
    def knapsack_heuristic(values, weights, capacity):
        """A heuristic for solving the 0/1 Knapsack Problem.
        Args:
            values: List of item values.
            weights: List of item weights.
            capacity: Maximum weight capacity of knapsack.
        Returns:
            Selected items and total value.
        """
        n = len(values)
        items = sorted(range(n), key=lambda i: values[i] / weights[i] if weights[i] > 0 else 0, reverse=True)

        total_value = 0
        total_weight = 0
        selected = []

        for i in items:
            if total_weight + weights[i] <= capacity:
                selected.append(i)
                total_value += values[i]
                total_weight += weights[i]

        return selected, total_value
    '''

    # Create evaluation
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program
    )

    # Create LLM
    llm = OpenAI(model='gpt-4')

    # Create and run MCTS-AHD
    mcts_ahd = MCTS_AHD(
        llm=llm,
        evaluation=evaluation,
        max_sample_nums=100,
        init_size=4,
        pop_size=10,
        selection_num=2,
        alpha=0.5,
        lambda_0=0.1
    )
    mcts_ahd.run()

Advanced Example with Profiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from llm4ad.base import Evaluation
    from llm4ad.method import MCTS_AHD
    from llm4ad.method.mcts_ahd import MAProfiler
    from llm4ad.tools.llm import OpenAI

    # Define your task and template
    task_description = "Design a heuristic for the Knapsack Problem..."
    template_program = '''
    def knapsack_heuristic(values, weights, capacity):
        """A heuristic for solving the 0/1 Knapsack Problem."""
        n = len(values)
        items = sorted(range(n), key=lambda i: values[i] / weights[i] if weights[i] > 0 else 0, reverse=True)
        total_value = 0
        total_weight = 0
        selected = []
        for i in items:
            if total_weight + weights[i] <= capacity:
                selected.append(i)
                total_value += values[i]
                total_weight += weights[i]
        return selected, total_value
    '''

    # Create evaluation with custom evaluator
    evaluation = Evaluation(
        task_description=task_description,
        template_program=template_program
    )

    # Create profiler to track results
    profiler = MAProfiler(
        log_dir='./logs/mcts_ahd_knapsack',
        initial_num_samples=0,
        log_style='complex'
    )

    # Create LLM
    llm = OpenAI(model='gpt-4')

    # Create and run MCTS-AHD with profiler
    mcts_ahd = MCTS_AHD(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        max_sample_nums=500,
        init_size=4,
        pop_size=10,
        selection_num=2,
        alpha=0.5,
        lambda_0=0.1,
        debug_mode=False,
        multi_thread_or_process_eval='thread'
    )
    mcts_ahd.run()

References
----------

- Zheng, Z., Xie, Z., Wang, Z., & Hooi, B. (2025). Monte carlo tree search for comprehensive exploration in llm-based automatic heuristic design. In Forty-first International Conference on Machine Learning (ICML), 2024.

- Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. "LLM4AD: A Platform for Algorithm Design with Large Language Model." arXiv preprint arXiv:2412.17287 (2024).
