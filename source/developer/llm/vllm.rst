.. _vllm_api:

vLLM API
========

The vLLM APIs provide an interface to deploy and interact with large language models
locally using the vLLM inference engine. vLLM offers high-throughput inference with
optimized serving.

Overview
--------

The LLM4AD platform provides two vLLM-related classes:

1. **LocalVLLMAPI**: High-level API for deploying multiple LLMs across multiple GPUs
2. **VLLMManager**: Utility class for managing vLLM model deployments

The vLLM backend provides:
- High-throughput inference using PagedAttention
- Multi-GPU support with tensor parallelism
- HTTP API server for inference requests

Setup Requirements
------------------

Before using vLLM APIs, ensure you have the following:

1. **Python Packages**: Install the required Python packages:

   .. code:: bash

       pip install vllm flask flask-cors transformers requests

2. **Hardware Requirements**:
   - NVIDIA GPU with CUDA 11.8+ or 12.1+
   - Sufficient GPU memory for the model (varies by model)
   - At least 16GB system RAM recommended

3. **Model Files**: Download your model (e.g., from Hugging Face):

   .. code:: bash

       # Example: Download Llama model
       # huggingface-cli download meta-llama/Llama-3.2-1B-Instruct

API Reference
-------------

Class Definition: VLLMManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: VLLMManager

   A manager class for deploying and controlling vLLM model instances.

Methods
~~~~~~~

.. method:: __init__(self)

   Initialize the VLLM manager.

   :return: A new VLLMManager instance.
   :rtype: VLLMManager

.. method:: deploy_models(self, model_path: str, tknz_path: str, gpus: List[int], ports: List[int], gpu_mem_utils: float | List[float] = None)

   Deploy vLLM models on specified GPUs.

   :param model_path: Path to the pretrained model directory.
   :type model_path: str
   :param tknz_path: Path to the tokenizer directory.
   :type tknz_path: str
   :param gpus: List of GPU indices to deploy models on.
   :type gpus: List[int]
   :param ports: List of HTTP ports for each model instance.
                 Must have the same length as ``gpus``.
   :type ports: List[int]
   :param gpu_mem_utils: GPU memory utilization ratio. Can be a single float
                         (applied to all GPUs) or a list of floats.
                         Defaults to 0.85 (85% memory usage).
   :type gpu_mem_utils: float | List[float], optional
   :raises ValueError: If ``len(gpus) != len(ports)``

   .. note::

      Each GPU will run a separate vLLM instance with its own HTTP server.
      The instances can handle inference requests independently.

.. method:: release_resources(self)

   Terminate all vLLM model processes and release GPU resources.

   This method:
   - Terminates all child processes
   - Clears CUDA cache
   - Frees GPU memory

   .. note::

      This is automatically called when the LocalVLLMAPI object is deleted.

.. method:: release_resources_(self)

   Legacy method for resource release. Use ``release_resources()`` instead.

Class Definition: LocalVLLMAPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: LocalVLLMAPI

   A concrete implementation of the LLM base class that uses vLLM for local inference.

Constructor
~~~~~~~~~~~

.. method:: __init__(self, model_path: str, tknz_path: str, gpus: List[int], ports: List[int], **kwargs)

   Initialize and deploy vLLM models on local GPUs.

   :param model_path: Path to the pretrained model directory.
   :type model_path: str
   :param tknz_path: Path to the tokenizer directory.
   :type tknz_path: str
   :param gpus: List of GPU indices to deploy models on.
                Each GPU will run one model instance.
   :type gpus: List[int]
   :param ports: List of HTTP ports for each model instance.
                 Must correspond to ``gpus`` one-to-one.
   :type ports: List[int]
   :param kwargs: Additional keyword arguments (passed to parent LLM class).

   .. note::

      This constructor automatically:
      1. Creates vLLM model instances on each specified GPU
      2. Starts HTTP servers on each specified port
      3. Creates a queue for load balancing across instances

   .. warning::

      Ensure no other service is using the specified ports.

Methods
~~~~~~~

.. method:: draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

   Generate a response from the vLLM model based on the provided prompt.

   :param prompt: The input prompt. Can be either:
                  - A string containing the user message
                  - A list of message dictionaries with 'role' and 'content' keys
   :type prompt: str | Any
   :param args: Additional positional arguments (unused).
   :param kwargs: Additional keyword arguments. Currently supports:
                  - ``temperature``: Sampling temperature (default: 1.0)
                  - ``top_p``: Nucleus sampling (default: 1.0)
                  - ``max_new_tokens``: Maximum tokens to generate (default: 4096)
   :return: The generated text content from the LLM response.
   :rtype: str

   .. note::

      The method automatically performs load balancing across multiple GPU instances.
      If one instance fails, it automatically retries with another instance.

.. method:: _do_request(self, content: str, url: str) -> str

   Internal method to make HTTP request to vLLM server.

   :param content: The prompt content.
   :type content: str
   :param url: The HTTP endpoint URL.
   :type url: str
   :return: The generated text.
   :rtype: str
   :raises Exception: If the request fails.

.. method:: close(self)

   Manually release resources. Called automatically when the object is deleted.

.. method:: __del__(self)

   Destructor that calls ``close()`` to ensure resources are released.

Examples
--------

Example 1: Basic Single-GPU Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_vllm import LocalVLLMAPI

    # Deploy model on single GPU
    sampler = LocalVLLMAPI(
        model_path="Llama-3.2-1B-Instruct",
        tknz_path="Llama-3.2-1B-Instruct",
        gpus=[0],
        ports=[22001]
    )

    # Generate response
    response = sampler.draw_sample("What is the meaning of life?")
    print(response)

    # Resources are automatically released when object is deleted
    del sampler

Example 2: Multi-GPU Deployment for Load Balancing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_vllm import LocalVLLMAPI

    # Deploy on multiple GPUs for higher throughput
    sampler = LocalVLLMAPI(
        model_path="Llama-3.2-1B-Instruct",
        tknz_path="Llama-3.2-1B-Instruct",
        gpus=[0, 1, 2, 3],        # Use 4 GPUs
        ports=[22001, 22002, 22003, 22004]  # Different port for each
    )

    # The sampler automatically balances requests across GPUs
    for i in range(10):
        response = sampler.draw_sample(f"Generate response number {i}")
        print(f"Response {i}: {response[:50]}...")

Example 3: Customizing GPU Memory Utilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_vllm import LocalVLLMAPI

    # Adjust memory utilization for each GPU
    sampler = LocalVLLMAPI(
        model_path="Llama-3.2-1B-Instruct",
        tknz_path="Llama-3.2-1B-Instruct",
        gpus=[0, 1],
        ports=[22001, 22002],
        gpu_mem_utils=[0.7, 0.9]  # Different utilization per GPU
    )

    # Or use a single value for all GPUs
    sampler = LocalVLLMAPI(
        model_path="Llama-3.2-1B-Instruct",
        tknz_path="Llama-3.2-1B-Instruct",
        gpus=[0],
        ports=[22001],
        gpu_mem_utils=0.5  # Use only 50% of GPU memory
    )

Example 4: Integration with LLM4AD Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_vllm import LocalVLLMAPI
    import llm4ad

    # Create vLLM sampler
    sampler = LocalVLLMAPI(
        model_path="Llama-3.2-1B-Instruct",
        tknz_path="Llama-3.2-1B-Instruct",
        gpus=[0],
        ports=[22001]
    )

    # Use with LLM4AD method
    task = llm4ad.tasks.optimization.SymbolicRegression(
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

    # Explicitly release resources
    sampler.close()

Example 5: Manual Resource Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from llm4ad.tools.llm.local_vllm import LocalVLLMAPI

    sampler = LocalVLLMAPI(
        model_path="Llama-3.2-1B-Instruct",
        tknz_path="Llama-3.2-1B-Instruct",
        gpus=[0],
        ports=[22001]
    )

    try:
        # Use the sampler
        response = sampler.draw_sample("Hello, world!")
        print(response)
    finally:
        # Always release resources explicitly
        sampler.close()

ed Models
---------

The vLLM backend supports various models including:

- **LLaMA Series**: LLaMA 2, LLaMA 3, LLaMA 3.2
- **Qwen Series**: Qwen 2, Qwen 3
- **Mistral**: Mistral 7B
- **Phi**: Phi-3

Make sure to use models that have been converted/pulled for vLLM format.

Common Issues and Troubleshooting
--------------------------------

1. **CUDA Out of Memory**: Reduce ``gpu_mem_utils`` or use a smaller model.

2. **Port Already in Use**: Change to a different port number.

3. **Model Loading Fails**: Ensure model path is correct and model is compatible with vLLM.

4. **Slow Inference**: Use tensor parallelism for larger models or optimize GPU settings.

5. **HTTP Connection Error**: Check firewall settings and ensure ports are accessible.

6. **Resources Not Released**: Always call ``close()`` or use context manager pattern.

See Also
--------

- :ref:`openai_api` - For OpenAI API integration
- :ref:`ollama_api` - For local Ollama deployments
- :ref:`https_api` - For custom HTTPS API implementations
