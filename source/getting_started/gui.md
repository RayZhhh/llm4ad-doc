# Graphical User Interface (GUI)

Welcome to the user documentation for our Graphical User Interface (GUI). The GUI provides a more intuitive and accessible way to use our platform and clearly displays results. 



## Main window

```{image} ../assets/gui_figs/gui_image.png
:width: 100%
```

The main window includes the following:

1. **Menu bar :**

The *Menu bar* contains four buttons, which, when clicked, will redirect to the [document](https://llm4ad-doc.readthedocs.io/en/latest/), [GitHub repository](https://github.com/Optima-CityU/LLM4AD), [website](http://www.llm4ad.com/index.html), and [QQ group](https://qm.qq.com/cgi-bin/qm/qr?k=4Imf8bn_d99-QXVcEJfOwCSD1KkcpbcD&jump_from=webapi&authKey=JtSmFh8BNKM97+TGnUdDgvT69TDTbo4UaLwgrZJSlsYqmVoCca/a5awU+TXt4zYB), respectively.

2. **Configuration panel:**

Users configure the settings of the large language model and set up the method and task to be executed in the *configuration panel*. 

3. **Results dashboard:**

The *Results dashboard* shows the best algorithm and objective value obtained in real-time.

4. **"Run" button:**

Click the *"Run" button* to execute the LLM4AD platform according to the setups in the *Configuration panel*.

5. **"Stop" button:**

Click the *"Stop" button* to terminate the execution process.

6. **"Log files" button:**

Click the *"Log files" button* to open the folder containing the log files.


## Execution

**Step 1**, clone the repository from GitHub and install all requirements (Please refer to [Installation](https://llm4ad-doc.readthedocs.io/en/latest/getting_started/installation.html)).

**Step 2**, execute the corresponding python script.

```
$ cd GUI
$ python run_gui.py
```

**Step 3**, set the parameter of the large language model.

- host, the ip of your API provider, no "https://", such as "api.bltcy.top".
- key, your API key which may start with "sk-......".
- model, the name of the large language model.

**Step 4**, select the **Method** to design the algorithm and set the parameter of the selected method.

**Step 5**, select which task you want to design an algorithm for. All tasks are divided into three types: `machine_learning`, `optimization`, and `science_discovery`. You can select the problem type in the Combobox.

```{image} ../assets/gui_figs/gui_combobox.png
:width: 40%
:align: center
```

**Step 6**, click the **Run** button. Results will be displayed in the `Results dashboard`.

```{image} ../assets/gui_figs/gui_gif.gif
:width: 80%
:align: center
```

## Adding new methods and tasks

**Step 1**, ensure that the code for new methods and tasks is placed in the correct folders:

- The code for methods should be stored in the `llm4ad/method` folder.
- Depending on the type of the problem, code for tasks should be stored in `llm4ad/task/machine_learning`, `llm4ad/task/optimization`, or `llm4ad/task/science_discovery`.

**Step 2**, configure through a YAML file named paras.yaml in the corresponding method/task folder to enable user-configurable parameters in the GUI. This file should contain all user-configurable parameters and their default values, following the specific format shown below:

```
name: [class name]
[parameter name]: [default value]
```

For example, in the case of the `eoh` method, the content of `llm4ad/method/eoh/paras.yaml` includes:

```
name: EoH
max_generations: 10
max_sample_nums: 20
pop_size: 4
num_evaluators: 4
```

The content indicates that users can configure the four parameters in the `eoh` method: `max_generations`, `max_sample_nums`, `pop_size`, and `num_evaluators`. These parameters default to 10, 20, 4, and 4, respectively.