# GUI Document

Welcome to the user documentation for our Graphical User Interface (GUI). The GUI provides a more intuitive and accessible way to use our platform and clearly displays results. 


## Main window

```{image} ./image.png
:width: 100%
```

The main window includes the following:

1. **Menu bar :**

The *Menu bar* contains four buttons, which, when clicked, will redirect to the document, GitHub repository, website, and QQ group, respectively.

2. **Configuration panel:**

The user configures the large language model and sets up the method and task to be executed in the *configuration panel*.

3. **Results dashboard:**

The *Results dashboard* shows the best algorithm and objective value obtained in real-time.

4. **"Run" button:**

Click the *"Run" button* to execute the LLM4AD platform according to the setups in the *Configuration panel*.

5. **"Stop" button:**

Click the *"Stop" button* to terminate the execution process.

6. **"Log files" button:**

Click the *"Stop" button* to open the folder containing the log files.





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

```{image} ./Combobox.png
:width: 40%
:align: center
```

**Step 6**, click the **Run** button. Results will be displayed in the `Results dashboard`.

```{image} ./gif.gif
:width: 80%
:align: center
```
