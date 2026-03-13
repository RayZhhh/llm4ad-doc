.. py:module:: llm4ad.base.code

========
Code Module
========

This module provides classes for representing and manipulating Python code as structured objects.

Overview
========

The code module defines three core classes:

* ``Function`` - A dataclass representing a parsed Python function with name, arguments, body, return type, and docstring.
* ``Program`` - A dataclass representing a complete Python program containing a preface (imports, global variables, classes) and a list of functions.
* ``TextFunctionProgramConverter`` - A utility class for converting between string representations and structured code objects.

These classes use Python's Abstract Syntax Tree (AST) parsing to decompose code into reusable components.

Function Class
==============

.. py:class:: Function

    A dataclass representing a parsed Python function. Contains all information needed about a function including name, arguments, body, return type, and optional docstring.

Attributes
----------

.. py:attribute:: algorithm

    :type: str
    :value: ''
    :annotation: Class-level attribute, typically used to identify the algorithm or task this function implements.

.. py:attribute:: name

    :type: str
    :annotation: The name of the function.

.. py:attribute:: args

    :type: str
    :annotation: The function arguments as a string (e.g., "a: np.ndarray, b: int = 5").

.. py:attribute:: body

    :type: str
    :annotation: The function body as a string, including indentation.

.. py:attribute:: return_type

    :type: str | None
    :annotation: The return type annotation (e.g., "np.ndarray"), or None if not specified.

.. py:attribute:: docstring

    :type: str | None
    :annotation: The function docstring, or None if not present.

.. py:attribute:: score

    :type: Any | None
    :annotation: A score assigned to this function (e.g., from evaluation). Initially None.

.. py:attribute:: evaluate_time

    :type: float | None
    :annotation: Time taken to evaluate this function in seconds. Initially None.

.. py:attribute:: sample_time

    :type: float | None
    :annotation: Time taken to generate this function in seconds. Initially None.

Methods
-------

.. py:method:: __str__(self) -> str

    Returns the string representation of the function.

    :returns: A string representation of the complete function including signature, docstring (if present), and body.
    :rtype: str

    Example:

    .. code-block:: python

        from llm4ad.base.code import Function

        func = Function(
            name='compute_sum',
            args='a: int, b: int',
            body='    return a + b',
            return_type='int'
        )
        print(str(func))
        # Output:
        # def compute_sum(a: int, b: int) -> int:
        #     return a + b


.. py:method:: __setattr__(self, name: str, value: str) -> None

    Sets an attribute with validation. Ensures no leading/trailing newlines in body and removes triple quotes from docstrings.

    :param name: The attribute name to set.
    :type name: str
    :param value: The value to assign.
    :type value: str


.. py:method:: __eq__(self, other: Function) -> bool

    Checks equality between two Function instances based on name, args, return_type, and body.

    :param other: The Function to compare with.
    :type other: Function
    :returns: True if all compared attributes are equal, False otherwise.
    :rtype: bool


Program Class
=============

.. py:class:: Program

    A frozen dataclass representing a parsed Python program. Contains a preface (imports, global variables, classes, etc.) and a list of Functions.

Attributes
----------

.. py:attribute:: preface

    :type: str
    :annotation: Everything from the beginning of the code up to but not including the first function definition. Includes imports, global variables, class definitions, etc.

.. py:attribute:: functions

    :type: list[Function]
    :annotation: A list of Function objects defined in the program.

Methods
-------

.. py:method:: __str__(self) -> str

    Returns the string representation of the complete program.

    :returns: A string representation including the preface and all functions.
    :rtype: str

    Example:

    .. code-block:: python

        from llm4ad.base.code import Program, Function

        program = Program(
            preface='import numpy as np\nWEIGHT = 10',
            functions=[
                Function(
                    name='func',
                    args='a: np.ndarray, b: np.ndarray',
                    body='    b = b + WEIGHT\n    return a + b',
                    return_type='np.ndarray'
                )
            ]
        )
        print(str(program))
        # Output:
        # import numpy as np
        # WEIGHT = 10
        #
        # def func(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        #     b = b + WEIGHT
        #     return a + b


.. py:method:: find_function_index(self, function_name: str) -> int

    Finds the index of a function by name in the program.

    :param function_name: The name of the function to find.
    :type function_name: str
    :returns: The index of the function in the functions list.
    :rtype: int
    :raises ValueError: If the function is not found or appears more than once.

    Example:

    .. code-block:: python

        from llm4ad.base.code import TextFunctionProgramConverter

        program_str = '''
        import numpy as np

        def func1(x):
            return x * 2

        def func2(x):
            return x + 1
        '''

        program = TextFunctionProgramConverter.text_to_program(program_str)
        index = program.find_function_index('func1')
        print(index)  # Output: 0


.. py:method:: get_function(self, function_name: str) -> Function

    Retrieves a function by name from the program.

    :param function_name: The name of the function to retrieve.
    :type function_name: str
    :returns: The Function object with the specified name.
    :rtype: Function
    :raises ValueError: If the function is not found or appears more than once.

    Example:

    .. code-block:: python

        from llm4ad.base.code import TextFunctionProgramConverter

        program_str = '''
        import numpy as np

        def target_func(a, b):
            return a + b
        '''

        program = TextFunctionProgramConverter.text_to_program(program_str)
        func = program.get_function('target_func')
        print(func.name)  # Output: target_func


.. py:method:: exec(self) -> List[Callable]

    Executes the program and returns callable functions.

    :returns: A list of callable functions defined in the program, in the order they appear.
    :rtype: List[Callable]

    Example:

    .. code-block:: python

        from llm4ad.base.code import TextFunctionProgramConverter

        program_str = '''
        def add(a, b):
            return a + b

        def multiply(a, b):
            return a * b
        '''

        program = TextFunctionProgramConverter.text_to_program(program_str)
        funcs = program.exec()
        print(funcs[0](2, 3))  # Output: 5
        print(funcs[1](2, 3))  # Output: 6


TextFunctionProgramConverter Class
===================================

.. py:class:: TextFunctionProgramConverter

    A utility class for converting between text (string representations of code) and structured code objects (Function and Program instances).

    This class provides static methods to parse Python code strings into structured objects and vice versa, enabling manipulation of code at the function level.

Methods
-------

.. py:method:: text_to_program(cls, program_str: str) -> Program | None

    Parses a Python code string into a Program object using AST.

    :param program_str: The Python code as a string.
    :type program_str: str
    :returns: A Program object representing the parsed code, or None if parsing fails.
    :rtype: Program | None

    Example:

    .. code-block:: python

        from llm4ad.base.code import TextFunctionProgramConverter

        code = '''
        import numpy as np

        def my_function(a: np.ndarray) -> np.ndarray:
            return a * 2
        '''

        program = TextFunctionProgramConverter.text_to_program(code)
        print(program.preface)  # Output: import numpy as np
        print(len(program.functions))  # Output: 1


.. py:method:: text_to_function(cls, program_str: str) -> Function | None

    Parses a Python code string containing exactly one function into a Function object.

    :param program_str: The Python code as a string containing a single function.
    :type program_str: str
    :returns: A Function object representing the parsed function, or None if parsing fails.
    :rtype: Function | None
    :raises ValueError: If the code contains more than one function.

    Example:

    .. code-block:: python

        from llm4ad.base.code import TextFunctionProgramConverter

        code = '''
        def calculate(a: int, b: int) -> int:
            return a + b
        '''

        func = TextFunctionProgramConverter.text_to_function(code)
        print(func.name)  # Output: calculate
        print(func.args)  # Output: a: int, b: int
        print(func.body)  # Output:     return a + b


.. py:method:: function_to_program(cls, function: str | Function, template_program: str | Program) -> Program | None

    Replaces the function body in a template program with a new function body.

    :param function: The function whose body will be used, either as a string or Function object.
    :type function: str | Function
    :param template_program: The template program containing the function to replace, either as a string or Program object.
    :type template_program: str | Program
    :returns: A new Program with the replaced function body, or None if conversion fails.
    :rtype: Program | None
    :raises ValueError: If the template program does not contain exactly one function.

    Example:

    .. code-block:: python

        from llm4ad.base.code import TextFunctionProgramConverter

        template = '''
        import numpy as np

        def target_func(a, b):
            """Template docstring."""
            return a
        '''

        new_body = '''
        result = a + b
        return result
        '''

        from llm4ad.base.code import Function
        new_func = Function(
            name='target_func',
            args='a, b',
            body='    result = a + b\n    return result'
        )

        program = TextFunctionProgramConverter.function_to_program(new_func, template)
        print(program.functions[0].body)
        # Output: result = a + b
        # return result


.. py:method:: program_to_function(cls, program: str | Program) -> Function | None

    Extracts the single function from a program.

    :param program: The program to extract the function from, either as a string or Program object.
    :type program: str | Program
    :returns: The Function object representing the single function in the program.
    :rtype: Function | None
    :raises ValueError: If the program does not contain exactly one function.

    Example:

    .. code-block:: python

        from llm4ad.base.code import TextFunctionProgramConverter

        code = '''
        import numpy as np

        def single_func(x):
            return x * 2
        '''

        func = TextFunctionProgramConverter.program_to_function(code)
        print(func.name)  # Output: single_func


Usage Examples
==============

Complete Example: Parsing and Modifying Code
--------------------------------------------

.. code-block:: python

    from llm4ad.base.code import (
        Function,
        Program,
        TextFunctionProgramConverter
    )

    # Example 1: Parse a string into Program and Function
    code_str = '''
    import numpy as np
    WEIGHT = 5

    def compute(a: np.ndarray, b: int) -> np.ndarray:
        """Compute weighted array."""
        return a * b + WEIGHT
    '''

    # Parse into Program
    program = TextFunctionProgramConverter.text_to_program(code_str)
    print(f"Preface: {program.preface}")
    print(f"Function count: {len(program.functions)}")

    # Get the function
    func = program.functions[0]
    print(f"Function name: {func.name}")
    print(f"Function args: {func.args}")
    print(f"Function body: {func.body}")
    print(f"Return type: {func.return_type}")

    # Example 2: Create a new Program from scratch
    new_program = Program(
        preface='import numpy as np\nimport math',
        functions=[
            Function(
                name='my_algorithm',
                args='data: np.ndarray, params: dict',
                body='    result = data.sum() * params["multiplier"]\n    return result',
                return_type='float'
            )
        ]
    )
    print(str(new_program))

    # Example 3: Replace function body using template
    template_str = '''
    import numpy as np

    def objective(x: np.ndarray) -> float:
        """Template docstring."""
        return 0.0
    '''

    new_body = '''
    return np.sum(x ** 2)
    '''

    # Create function with new body
    func = Function(
        name='objective',
        args='x: np.ndarray',
        body='    return np.sum(x ** 2)',
        return_type='float'
    )

    # Replace in template
    result_program = TextFunctionProgramConverter.function_to_program(func, template_str)
    print(str(result_program))


Complete Example: Executing Generated Code
-------------------------------------------

.. code-block:: python

    from llm4ad.base.code import TextFunctionProgramConverter

    # Define a simple algorithm
    algo_code = '''
    def sort_list(arr):
        return sorted(arr)
    '''

    # Parse and execute
    program = TextFunctionProgramConverter.text_to_program(algo_code)
    callable_funcs = program.exec()

    # Call the function
    result = callable_funcs[0]([3, 1, 4, 1, 5, 9, 2, 6])
    print(result)  # Output: [1, 1, 2, 3, 4, 5, 6, 9]
