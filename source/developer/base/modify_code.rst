.. py:module:: llm4ad.base.modify_code

============
ModifyCode Module
============

Overview
========

The modify_code module provides utilities for programmatically modifying Python source code. The ``ModifyCode`` class offers static methods for:

- Adding decorators to functions
- Adding import statements
- Adding random seeds for reproducibility
- Replacing division with protected division
- Renaming functions
- Analyzing function usage in code

This module uses Python's Abstract Syntax Tree (AST) for robust code manipulation.

ModifyCode Class
=================

.. py:class:: ModifyCode

    A utility class with static methods for modifying Python source code.

    All methods are class methods and can be called directly without instantiation.

Methods
-------

.. py:method:: add_decorator(cls, program: str, function_name: str, decorator_name: str | List[str], decorator_args: List[str | Tuple[str, Any]] = None) -> str

    Adds a decorator to a specified function in the program.

    :param program: The program code as a string.
    :type program: str
    :param function_name: The name of the function to decorate.
    :type function_name: str
    :param decorator_name: The name of the decorator. Can be a string (e.g., 'numba.jit', 'torch.jit.script') or a list of strings (e.g., ['numba', 'jit']).
    :type decorator_name: str | List[str]
    :param decorator_args: Optional arguments for the decorator as a list. Can include positional arguments (as strings) or keyword arguments (as tuples of (name, value)). Default is None.
    :type decorator_args: List[str | Tuple[str, Any]] | None
    :returns: The modified program string with the decorator added.
    :rtype: str

    Example 1 - Simple Decorator:

    .. code-block:: python

        program = '''
        def f():
            return 0'''

        result = ModifyCode.add_decorator(program, 'f', 'torch.jit.script')
        # Result:
        # @torch.jit.script()
        # def f():
        #     return 0

    Example 2 - Decorator with kwargs:

    .. code-block:: python

        program = '''
        def f():
            return 0'''

        result = ModifyCode.add_decorator(program, 'f', ['numba', 'jit'], [('nopython', True)])
        # Result:
        # @numba.jit(nopython=True)
        # def f():
        #     return 0

    Example 3 - Complex Decorator:

    .. code-block:: python

        program = '''
        def f():
            return 0'''

        result = ModifyCode.add_decorator(program, 'f', 'a.b.c.d', [1, True, ('e', 'all'), ('f', True)])
        # Result:
        # @a.b.c.d(1, True, e='all', f=True)
        # def f():
        #     return 0


.. py:method:: add_import_package_statement(cls, program: str, package_name: str, as_name: str | None = None, *, check_imported: bool = True) -> str

    Adds an import statement to the program.

    :param program: The program code as a string.
    :type program: str
    :param package_name: The name of the package to import.
    :type package_name: str
    :param as_name: Optional alias for the imported package (e.g., 'np' for 'numpy'). Default is None.
    :type as_name: str | None
    :param check_imported: If True, checks if the package is already imported and returns the original program if so. Default is True.
    :type check_imported: bool
    :returns: The modified program string with the import added.
    :rtype: str

    Example:

    .. code-block:: python

        program = '''
        def f():
            return np.array([1, 2, 3])
        '''

        result = ModifyCode.add_import_package_statement(program, 'numpy', 'np')
        # Result:
        # import numpy as np
        #
        # def f():
        #     return np.array([1, 2, 3])


.. py:method:: add_numpy_random_seed_to_func(cls, program: str, func_name: str, seed: int = 2024) -> str

    Adds a numpy random seed assignment at the beginning of a function.

    :param program: The program code as a string.
    :type program: str
    :param func_name: The name of the function to add the seed to.
    :type func_name: str
    :param seed: The random seed value. Default is 2024.
    :type seed: int
    :returns: The modified program string with the seed added.
    :rtype: str

    Example:

    .. code-block:: python

        program = '''
        def compute():
            a = np.random.random()
            return a * 2
        '''

        result = ModifyCode.add_numpy_random_seed_to_func(program, 'compute', 42)
        # Result:
        # def compute():
        #     np.random.seed(42)
        #     a = np.random.random()
        #     return a * 2


.. py:method:: replace_div_with_protected_div(cls, program: str, delta: float = 1e-5, numba_accelerate: bool = False, return_div_func_name: bool = False) -> str | Tuple[str, str]

    Replaces division operations with a protected division function that adds a small delta to the denominator.

    :param program: The program code as a string.
    :type program: str
    :param delta: The delta value to add to the denominator. Default is 1e-5.
    :type delta: float
    :param numba_accelerate: If True, adds the numba jit decorator to the protected division function. Default is False.
    :type numba_accelerate: bool
    :param return_div_func_name: If True, returns a tuple of (modified_program, function_name). Default is False.
    :type return_div_func_name: bool
    :returns: The modified program string with protected division, or a tuple if return_div_func_name is True.
    :rtype: str | Tuple[str, str]

    Example:

    .. code-block:: python

        program = '''
        def compute(a, b):
            return a / b
        '''

        result = ModifyCode.replace_div_with_protected_div(program, delta=1e-8)
        # Result:
        # def compute(a, b):
        #     return _protected_div(a, b)
        #
        #
        # def _protected_div(x, y, delta=1e-08):
        #     return x / (y + delta)


.. py:method:: add_np_random_seed_below_numpy_import(cls, program: str, seed: int = 2024) -> str

    Adds numpy import (if needed) and inserts np.random.seed() call below the import statement.

    :param program: The program code as a string.
    :type program: str
    :param seed: The random seed value. Default is 2024.
    :type seed: int
    :returns: The modified program string with the import and seed.
    :rtype: str

    Example:

    .. code-block:: python

        program = '''
        import numpy as np

        def f():
            return np.random.random()
        '''

        result = ModifyCode.add_np_random_seed_below_numpy_import(program, 123)
        # Result:
        # import numpy as np
        # np.random.seed(123)
        #
        # def f():
        #     return np.random.random()


.. py:method:: add_numba_decorator(cls, program: str, function_name: str | List[str]) -> str

    Adds the @numba.jit(nopython=True) decorator to one or more functions.

    :param program: The program code as a string.
    :type program: str
    :param function_name: The name of the function to decorate, or a list of function names.
    :type function_name: str | List[str]
    :returns: The modified program string with the numba decorator added.
    :rtype: str

    Example:

    .. code-block:: python

        program = '''
        import numba

        def func(a: np.ndarray):
            return a * 2
        '''

        result = ModifyCode.add_numba_decorator(program, 'func')
        # Result:
        # import numba
        #
        # @numba.jit(nopython=True)
        # def func(a: np.ndarray):
        #     return a * 2

    Note:
        Not all numpy functions support numba acceleration (e.g., np.piecewise).


.. py:method:: rename_function(cls, code: str, source_name: str, target_name: str) -> str

    Renames function calls from source_name to target_name.

    :param code: The program code as a string.
    :type code: str
    :param source_name: The original function name to replace.
    :type source_name: str
    :param target_name: The new function name.
    :type target_name: str
    :returns: The modified program string with renamed function calls.
    :rtype: str

    Example:

    .. code-block:: python

        code = '''
        def old_func(x):
            return x + 1

        result = old_func(5)
        '''

        result = ModifyCode.rename_function(code, 'old_func', 'new_func')
        # Result:
        # def new_func(x):
        #     return x + 1
        #
        # result = new_func(5)


.. py:method:: get_functions_name(cls, code: str) -> MutableSet[str]

    Returns the set of all function names that are called in the code.

    :param code: The program code as a string.
    :type code: str
    :returns: A set of function names found in the code.
    :rtype: MutableSet[str]

    Example:

    .. code-block:: python

        code = '''
        import numpy as np

        def my_func(x):
            return np.sum(x) + len(x)

        result = my_func(arr)
        another = len(result)
        '''

        functions = ModifyCode.get_functions_name(code)
        print(functions)
        # Output: {'np', 'sum', 'len', 'my_func'}


.. py:method:: yield_decorated(cls, code: str, module: str, name: str) -> Iterator[str]

    Yields names of functions decorated with a specific decorator.

    :param code: The program code as a string.
    :type code: str
    :param module: The module name in the decorator (e.g., 'numba').
    :type module: str
    :param name: The decorator name (e.g., 'jit').
    :type name: str
    :returns: An iterator of function names that have the specified decorator.
    :rtype: Iterator[str]

    Example:

    .. code-block:: python

        code = '''
        import numba
        import torch

        @numba.jit
        def fast_func(x):
            return x * 2

        @torch.jit.script
        def scripted_func(x):
            return x + 1

        def normal_func(x):
            return x - 1
        '''

        decorated = list(ModifyCode.yield_decorated(code, 'numba', 'jit'))
        print(decorated)
        # Output: ['fast_func']


Usage Examples
==============

Example: Preparing Code for Evaluation
--------------------------------------

.. code-block:: python

    from llm4ad.base.modify_code import ModifyCode

    # Original candidate code
    code = '''
    import numpy as np

    def objective(x):
        return np.sum(x ** 2) / len(x)
    '''

    # Step 1: Add numba decorator for acceleration
    code = ModifyCode.add_numba_decorator(code, 'objective')

    # Step 2: Replace division with protected division
    code = ModifyCode.replace_div_with_protected_div(code, delta=1e-8, numba_accelerate=True)

    # Step 3: Add random seed for reproducibility
    code = ModifyCode.add_numpy_random_seed_to_func(code, 'objective', seed=42)

    print(code)
    # Output:
    # import numpy as np
    # import numba
    #
    # @numba.jit(nopython=True)
    # def objective(x):
    #     np.random.seed(42)
    #     return _protected_div(np.sum(x ** 2), len(x))
    #
    #
    # def _protected_div(x, y, delta=1e-08):
    #     return x / (y + delta)


Example: Complete Code Preparation Pipeline
--------------------------------------------

.. code-block:: python

    from llm4ad.base.modify_code import ModifyCode

    def prepare_for_evaluation(code: str, func_name: str, seed: int = 42) -> str:
        """Prepare code for evaluation with numba, protected div, and random seed."""

        # Ensure numpy is imported
        code = ModifyCode.add_import_package_statement(code, 'numpy', 'np')

        # Add numba decorator
        code = ModifyCode.add_numba_decorator(code, func_name)

        # Replace division with protected version
        code = ModifyCode.replace_div_with_protected_div(code, delta=1e-8, numba_accelerate=True)

        # Add random seed
        code = ModifyCode.add_numpy_random_seed_to_func(code, func_name, seed)

        return code


    original = '''
    def compute(arr):
        return sum(arr) / len(arr)
    '''

    prepared = prepare_for_evaluation(original, 'compute', seed=123)
    print(prepared)


Example: Analyzing Code
-----------------------

.. code-block:: python

    from llm4ad.base.modify_code import ModifyCode

    code = '''
    import numpy as np

    @numba.jit(nopython=True)
    def fast_compute(data):
        result = 0
        for i in range(len(data)):
            result += data[i] * 2
        return result

    def slow_compute(data):
        return [x * 2 for x in data]

    output = fast_compute(arr)
    count = len(output)
    '''

    # Get all function calls
    functions = ModifyCode.get_functions_name(code)
    print(f"Function calls: {functions}")

    # Find numba-decorated functions
    decorated = list(ModifyCode.yield_decorated(code, 'numba', 'jit'))
    print(f"Numba-decorated: {decorated}")


Example: Transforming Old API to New API
----------------------------------------

.. code-block:: python

    from llm4ad.base.modify_code import ModifyCode

    # Old code using deprecated function
    old_code = '''
    def process(data):
        return old_transform(data, method='fast')
    '''

    # Replace old function with new
    new_code = ModifyCode.rename_function(old_code, 'old_transform', 'new_transform')
    print(new_code)
    # Output:
    # def process(data):
    #     return new_transform(data, method='fast')


Example: Combining Multiple Modifications
-----------------------------------------

.. code-block:: python

    from llm4ad.base.modify_code import ModifyCode

    def enhance_function(code: str, func_name: str, decorator: str = None) -> str:
        """Apply multiple enhancements to a function."""

        # Add custom decorator if specified
        if decorator:
            code = ModifyCode.add_decorator(code, func_name, decorator)

        # Add numpy import if needed
        code = ModifyCode.add_import_package_statement(code, 'numpy', 'np')

        # Replace division with protected version
        code = ModifyCode.replace_div_with_protected_div(code, delta=1e-10)

        return code


    original = '''
    def calculate(x, y):
        return x / y
    '''

    enhanced = enhance_function(original, 'calculate', 'functools.lru_cache')
    print(enhanced)
