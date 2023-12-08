Evolving Biologically Plausible Recurrent Neural Networks for Temporal Prediction
===============================================

About
-----
This is the code for my thesis project for the Norwegian University of Science and Technology, written at the University of Tokyo. The code extends the NEAT and ES-HyperNEAT implementations found in the [Pureples](https://github.com/ukuleleplayer/pureples) and [NEAT-Python](https://github.com/CodeReclaimers/neat-python) libraries, briefly explained below.

**NEAT** (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks.  
**HyperNEAT** (Hypercube-based NEAT) is a method developed by Kenneth O. Stanley utilizing NEAT. It is a technique for evolving large-scale neural networks using the geometric regularities of the task domain.  
**ES-HyperNEAT** (Evolvable-substrate HyperNEAT) is a method developed by Sebastian Risi and Kenneth O. Stanley utilizing HyperNEAT. It is a technique for evolving large-scale neural networks using the geometric regularities of the task domain. In contrast to HyperNEAT, the substrate used during evolution is able to evolve. This rids the user of some initial work and often creates a more suitable substrate.

Getting started
---------------
This section briefly describes how to install and run experiments.  

### Installation Guide
First, make sure you have the dependencies installed: `numpy`, `neat-python`, `graphviz`, `matplotlib` and `gym`.  
All the above can be installed using [pip](https://pip.pypa.io/en/stable/installing/).  
Next, download the source code and run `setup.py` (`pip install .`) from the root folder.

| Argument              | Default                                              | Effect                                                                                                                                                                                                                                                                                                                                                 |
| --------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| gens                  | 1                                                    | The number of generations to run NEAT for                                                                                                                                                                                                                                                                                                              |
| target_folder         | None                                                 | The target folder of the experiment results                                                                                                                                                                                                                                                                                                            |
| suffix                | ""                                                   | Suffix to append to the folder name, is overwritten by the folder argument                                                                                                                                                                                                                                                                             |
| overwrite             | False                                                | Whether or not to overwrite the target folder if it already exists.                                                                                                                                                                                                                                                                                    |
| load                  | None                                                 | Path to the .pkl file of a genome to load                                                                                                                                                                                                                                                                                                              |
| config                | "pureples/experiments/ready_go/config_neat_ready_go" | Path to the config file used when running                                                                                                                                                                                                                                                                                                              |
| hebbian_type          | "positive"                                           | Affects how hebbian updates are applied to the networks. "positive" only lets positive connections be affected by hebbian updates. "signed" lets both positive and negative connections be affected by hebbian updates, but ensures the sign of each connection stays the same. "unsigned" does not ensure the sign of each connection stays the same. |
| firing_threshold      | 0.20                                                 | The minimum node activity required for a node to be considered firing                                                                                                                                                                                                                                                                                  |
| hebbian_learning_rate | 0.05                                                 | Affects the magnitude of hebbian updates and holdover of previous values                                                                                                                                                                                                                                                                               |
| binary_weights        | False                                                | If the network weights are binary (only taking values -1 or 1) or not                                                                                                                                                                                                                                                                                  |
| experiment            | foreperiod                                           | Which experiment to run NEAT for                                                                                                                                                                                                                                                                                                                       |
| max_foreperiod        | 25                                                   | The maximum length of the foreperiod, given in milliseconds.                                                                                                                                                                                                                                                                                           |
| trial_delay_range     | [0,3]                                                | The range of random delay between each trial, given in timesteps. Each timestep is 5 milliseconds long.                                                                                                                                                                                                                                                |
| foreperiods           | [1,2,3,4,5]                                          | The foreperiods the networks are trained on, given in timesteps.                                                                                                                                                                                                                                                                                       |
| ordering              | []                                                   | The ordering, if any, of the foreperiods. Defaults to random ordering if empty.                                                                                                                                                                                                                                                                        |
| flip_pad_data         | True                                                 | Whether or not to append the flipped ordering of the foreperiod blocks to training set, in order to ensure more consistent results.                                                                                                                                                                                                                    |
| end_test              | 0                                                    | The amount of extra tests to run at the end. Currently only the omission test is hardcoded to run if end_test > 0.                                                                                                                                                                                                                                     |
| reset                 | False                                                | Whether or not to reset the network between foreperiod blocks                                                                                                                                                                                                                                                                                          |
| model                 | "rnn"                                                | Which network model to use when running NEAT. "rnn" is a standard Recurrent Neural Networks. "iznn" is an Izhikevich Spiking Neural Network. "rnn_d" is a Recurrent Neural Network activity holdover between trials.                                                                                                                                   |


### Organization
The code used for running the different versions of NEAT used in the thesis are found in [pureples/experiments/ready_go](pureples/experiments/ready_go).
Within this folder you also find the [results](pureples/experiments/ready_go/results) folder containing the experiment results, while [thesis_networks](pureples/experiments/ready_go/thesis_networks/) contains the specific networks used in the thesis (net-a, b, c2 and d).

Code shared between the different versions of NEAT, as well as the Ready-Go experiment, are found in [pureples/shared](pureples/shared/).

### Running Networks
You can run your chosen NEAT variant through the terminal, though I recommend [hebbian_neat_ready_go.py](pureples/experiments/ready_go/hebbian_neat_ready_go.py), as that is the most up to date. 

A nunmber of command-line arguments will modify the behaviour of NEAT, in addition to the config files.


### Experimenting
How to experiment using NEAT will not be described, since this is the responsibility of the `neat-python` library.

Setting up an experiment for **HyperNEAT**:
* Define a substrate with input nodes and output nodes as a list of tuples. The hidden nodes is a list of lists of tuples where the inner lists represent layers. The first list is the topmost layer, the last the bottommost.
* Create a configuration file defining various NEAT specific parameters which are used for the CPPN.
* Define a fitness function setting the fitness of each genome. This is where the CPPN and the ANN is constructed for each generation - use the `create_phenotype_network` method from the `hyperneat` module.
* Create a population with the configuration file made in (2).
* Run the population with the fitness function made in (3) and the configuration file made in (2). The output is the genome solving the task or the one closest to solving it.

Setting up an experiment for **ES-HyperNEAT**:
Use the same setup as HyperNEAT except for:
* Not declaring hidden nodes when defining the substrate.
* Declaring ES-HyperNEAT specific parameters.
* Using the `create_phenotype_network` method residing in the `es_hyperneat` module when creating the ANN.

If one is trying to solve an experiment defined by the [OpenAI Gym](https://gym.openai.com/) it is even easier to experiment. In the `shared` module a file called `gym_runner` is able to do most of the work. Given the number of generations, the environment to run, a configuration file, and a substrate, the relevant runner will take care of everything regarding population, fitness function etc.

Please refer to the sample experiments included for further details on experimenting. 

