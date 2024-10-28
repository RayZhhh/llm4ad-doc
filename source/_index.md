# Project Structure



```{rst} 
.. figure:: ./assets/figs/project_structure.png
    :alt: llm-eps
    :align: center
    :width: 100%
```




## 🛠️ Project structure

|----**[alevo]**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[base]** basic package for modifying code, secure evaluation, profiling experiments.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`code.py` implements two classes, Function and Program, to record the evolved heuristic.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`evaluate.py` implements evaluate interface for user, and implements the secure evaluator.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`evaluate_multi_program.py` a more generic interface and secure evaluator to evaluate multiple programs simultaneously.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`modify_code.py` implements methods two modify heuristic's code using ast.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`sample.py` implements sampler interface for the user, and also response content trimmer.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[methods]** package for implementation classes of various LLM-EPS methods.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[tools]**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[llm]** package for examples to use local LLMs, and use API interfaces.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[profiler]** package for base implementaions of Tensorboard and WandB profiler (logger).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;~~|----**[tasks]** package for AHD task examples (to be updated).~~

## ⚙️ Usage of this package

### Use [base] package to modify code and perform secure evaluation for code

The `base` package is the basic package for all methods, which provides **useful tools** for manipulating the code for Python programs and functions, and methods to extract the valid part (Python code) from LLM-generated contents. Please
refer to the tutorials (jupyter notebooks) in the `tutorials_for_base_package` folder.

You are encouraged to implement your own EPS method using this package!

### Use [method] package to perform an AHD task

We provide LLM-free examples for you. Please look for examples in `example/online_bin_packing/fake_xxx.py`. These examples provide a "fake sampler" that randomly selects code from the database to simulate the sampling process, which can
help you test and debug our pipeline more easily.

Run in terminal:

```shell
cd example/online_bin_packing
python fake_funsearch.py
```

We also provide tutorials for procedures to customize your own AHD tasks. An example of online bin packing problems (LLM-free or using your API key) is demonstrated on
Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayZhhh/py-alevo/blob/main/online_bin_packing/online_bin_packing_tutorial.ipynb).

## 🔗 Citation

Very much apprciate if you could cite our works if you get help from this package!!!

```la
@inproceedings{fei2024eoh,
    title={Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model},
    author={Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, Qingfu Zhang},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2024},
    url={https://arxiv.org/abs/2401.02051}
}

@inproceedings{zhang2024understanding,
    title={Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models}, 
    author={Rui Zhang and Fei Liu and Xi Lin and Zhenkun Wang and Zhichao Lu and Qingfu Zhang},
    booktitle={International Conference on Parallel Problem Solving From Nature},
    year={2024},
    url={https://arxiv.org/abs/2407.10873}, 
}
```

## ❌ Issues

If you encounter any difficulty using the code, please do not hesitate to submit an issue!

