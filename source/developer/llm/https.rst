.. _https_api:

HTTPS API
=========

The HttpsApi class provides a generic interface for interacting with LLM APIs over HTTPS.
It supports any OpenAI-compatible API endpoint that uses bearer token authentication.

Overview
--------

The ``HttpsApi`` class extends the base ``LLM`` class and provides a flexible way to connect
to various LLM API providers through HTTPS. It implements the OpenAI Chat Completions API
format, making it compatible with any provider that follows this standard.

Key Features:
- Generic HTTPS-based API calls
- Bearer token authentication
- Configurable request parameters (temperature, max_tokens, top_p)
- Built-in error handling with retry logic
- Debug mode for detailed error reporting

Setup Requirements
------------------

Before using HttpsApi, ensure you have the following:

1. **API Endpoint**: A valid HTTPS endpoint that supports the OpenAI Chat Completions API format.

2. **API Key**: A valid API key for authentication.

3. **Model Name**: The model identifier as recognized by your API provider.

API Reference
-------------

Class Definition
~~~~~~~~~~~~~~~~

.. class:: HttpsApi

   A concrete implementation of the LLM base class that interfaces with HTTPS-based LLM APIs.

Constructor
~~~~~~~~~~

.. method:: __init__(self, host: str, key: str, model: str, timeout: int = 60, **kwargs)

   Initialize the HTTPS API client.

   :param host: The hostname of the API endpoint. Note: Do NOT include ``https://`` prefix.
                Example: ``"api.openai.com"`` or ``"api.example.com"``
   :type host: str
   :param key: The API key for authentication. This will be used in the Bearer token.
   :type key: str
   :param model: The LLM model identifier to use (e.g., ``"gpt-4"``, ``"gpt-3.5-turbo"``).
   :type model: str
   :param timeout: Request timeout in seconds. Defaults to 60.
   :type timeout: int, optional
   :param kwargs: Additional parameters for the API request:
                  - ``max_tokens``: Maximum tokens to generate (default: 4096)
                  - ``top_p``: Nucleus sampling parameter (default: None)
                  - ``temperature``: Sampling temperature (default: 1.0)
   :type kwargs: Any, optional

   .. warning::

      The host parameter should NOT include the ``https://`` prefix. The class
      automatically establishes an HTTPS connection.

   .. note::

      The User-Agent is set to ``"Apifox/1.0.0 (https://apifox.com)"`` by default.

Methods
~~~~~~~

.. method:: draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

   Generate a response from the LLM API based on the provided prompt.

   :param prompt: The input prompt. Can be either:
                  - A string containing the user message
                  - A list of message dictionaries with 'role' and 'content' keys
   :type prompt: str | Any
   :param args: Additional positional arguments (unused).
   :param kwargs: Additional keyword arguments to override default request parameters.
                  Can include ``max_tokens``, ``top_p``, ``temperature``, etc.
   :return: The generated text content from the LLM response.
   :rtype: str

   :raises RuntimeError: After 10 consecutive errors when in debug mode.

   .. note::

      - When passing a string prompt, it is automatically wrapped in a user message format
      - The method has built-in retry logic: it will retry indefinitely (in non-debug mode)
        or up to 10 times (in debug mode) on failure
      - The response is automatically extracted from the OpenAI-compatible format

   .. note::

      Inherits ``debug_mode`` from the parent LLM class. When ``debug_mode=True``,
      errors are accumulated and a RuntimeError is raised after 10 consecutive failures.

Examples
--------

Example 1: Using OpenAI API via HTTPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_https import HttpsApi

    # Initialize with OpenAI API credentials
    sampler = HttpsApi(
        host="api.openai.com",
        key="your-openai-api-key",
        model="gpt-4",
        timeout=60
    )

    # Generate a response
    response = sampler.draw_sample("What is the capital of Japan?")
    print(response)  # Output: Tokyo

Example 2: Using Third-Party API Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_https import HttpsApi

    # Using an alternative API provider
    sampler = HttpsApi(
        host="api.another-provider.com",  # No https:// prefix
        key="your-provider-api-key",
        model="gpt-3.5-turbo",
        timeout=30
    )

    response = sampler.draw_sample("Explain quantum computing in simple terms.")

Example 3: Customizing Request Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_https import HttpsApi

    # Initialize with default parameters
    sampler = HttpsApi(
        host="api.openai.com",
        key="your-api-key",
        model="gpt-4",
        timeout=60,
        max_tokens=2000,      # Max tokens in response
        temperature=0.7,      # Creativity level (0.0 to 2.0)
        top_p=0.9            # Nucleus sampling
    )

    # Override parameters per request
    response = sampler.draw_sample(
        prompt="Write a short story about AI.",
        temperature=0.9,      # Override default temperature
        max_tokens=1000      # Override default max_tokens
    )

Example 4: Using Chat Message Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_https import HttpsApi

    sampler = HttpsApi(
        host="api.openai.com",
        key="your-api-key",
        model="gpt-4"
    )

    # Using multi-turn conversation format
    messages = [
        {"role": "system", "content": "You are a helpful Python programming assistant."},
        {"role": "user", "content": "How do I sort a dictionary by value in Python?"},
        {"role": "assistant", "content": "You can use the sorted() function with a lambda..."},
        {"role": "user", "content": "Can you show me an example?"}
    ]

    response = sampler.draw_sample(messages)
    print(response)

Example 5: Integration with LLM4AD Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_https import HttpsApi
    import llm4ad

    # Create HTTPS API sampler
    sampler = HttpsApi(
        host="api.openai.com",
        key="your-api-key",
        model="gpt-4",
        timeout=60,
        temperature=0.8
    )

    # Use with an LLM4AD method
    task = llm4ad.tasks.optimization.SymReg(
        dimension=5,
        num_samples=100,
        eval_budget=1000
    )

    method = llm4ad.methods.eoh.EoH(
        task=task,
        sampler=sampler,
        num_iterations=50
    )

    result = method.run()
    print(f"Best solution: {result.best_solution}")
    print(f"Best fitness: {result.best_fitness}")

Example 6: Using Debug Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.llm_api_https import HttpsApi

    sampler = HttpsApi(
        host="api.openai.com",
        key="your-api-key",
        model="gpt-4"
    )

    # Enable debug mode for detailed error reporting
    sampler.debug_mode = True

    try:
        response = sampler.draw_sample("Hello!")
        print(response)
    except RuntimeError as e:
        print(f"Error: {e}")
        # Check API key and host configuration

Compatible Providers
--------------------

The HttpsApi class is compatible with any provider that supports the OpenAI
Chat Completions API format, including:

- **OpenAI**: ``api.openai.com`` - GPT-4, GPT-3.5 Turbo
- **Azure OpenAI**: ``<your-resource>.openai.azure.com``
- **Anthropic** (via compatible proxy): Various endpoints
- **Local LLM Servers**: Local deployments with OpenAI-compatible API
- **Third-Party Aggregators**: Various OpenAI-compatible services

Common Issues and Troubleshooting
--------------------------------

1. **Connection Refused**: Check that the host is correct and the service is running.

2. **Authentication Error (401)**: Verify your API key is correct and has not expired.

3. **Rate Limiting (429)**: Implement exponential backoff or reduce request frequency.

4. **Invalid Request (400)**: Check model name and parameter values are valid.

5. **Timeout Issues**: Increase the timeout parameter for slower connections.

6. **SSL/TLS Errors**: Ensure your environment has proper SSL certificates installed.

7. **Host Format**: Remember NOT to include ``https://`` in the host parameter.

See Also
--------

- :ref:`openai_api` - For OpenAI Python SDK-based integration
- :ref:`ollama_api` - For local Ollama deployments
- :ref:`vllm_api` - For local vLLM deployments
