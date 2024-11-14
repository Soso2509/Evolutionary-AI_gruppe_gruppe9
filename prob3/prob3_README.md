# Running code from Problem 3
**Make sure you are in the root folder `\Evolutionary-AI_gruppe_gruppe9`**

## Overview
0. [Dataset info](#information-regarding-data-set)
1. [Ant Colony Optimization](#1-run-ant-colony-optimization-aco)
2. [Particle Swarm Optimization](#2-run-particle-swarm-optimization-pso)
3. [Route Feasibility Checker (OPTIONAL)](#route-feasibility-checker-optional)


## Information regarding data set
To make the data easier to work with when implementing the optimisation algorithms we wrote a little code to convert from the .txt file that was downloaded to a .csv file that is easier to work with.

To run it with the "c101"-dataset the conversion has already been done and data is saved in the `data.csv` file, but if someone wants to run the algorithms with another dataset the data needs to be pasted into the `cXXX_to_csv.py` file in the format of a list of lists.\
Example:
``` csv
variable_name = [
[0, 40, 50, 0, 0, 1236, 0],
[1, 45, 68, 10, 912, 967, 90],
]
```
Make sure the data matches with the header `['CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME']` and that the name of the correct variable is in the function
``` python
df = pd.DataFrame(variable_name, columns=header)
```
Then the code can be run and det `data.csv` will be overvritten with the new dataset
``` bash
python .\prob3\1.prob3_data\cXXX_to_csv.py
```
## 1. Run Ant Colony Optimization (ACO)
For each ant and iteration it will print the total cost of the routes, and the cost if the current overall best collection routes found.

Lastly will print the best found vehicle routes in the terminal and the total cost of it.\
Also shows a visualisation of the routes taken by each vehicle.
```bash
python .\prob3\2.prob3_kode\aco_program.py
```

## 2. Run Particle Swarm Optimization (PSO)
Will print the best found vehicle routes in the terminal and the total cost of it.\
Also shows a visualisation of the routes taken by each vehicle.
```bash
python .\prob3\2.prob3_kode\pso_program.py
```

## Route Feasibility Checker (OPTIONAL)
To double check that the best routes found by the algorithms is feasible an "Route Feasibility Checker" was written

### Preparation of Data
The routes printed by the algorithms are sadly needs a bit of an adjustment for the feasibility checker.\
To have the code work properly the data needs to be formatted as a list of lists, `[[],[],[]]`

#### Ant Colony Optimisation
For ACO there is only a need to remove all occurences of `np.int64(` and `)`\
VScodes **"Change all occurrences"** is very helpful here

#### Particle Swarm Optimisation
With PSO it is a bit more that need to be removed and added\
VScodes **"Change all occurrences"** is once again very helpful
- Remove all occurrences of `Vehicle \<number> Route (sorted by time):`
- Remove all occurences of `np.int64(` and `)`
- After each `]` add a `,`
- After each `[` add a `0,`
- Before each `]` add a `,0`
- Lastly encase the list within `[]`

### Add data
The list of list need to be pasted to the `routes` variable
```python
routes =
```

### Run code
As you run the code there is an output pr route, either a\
"*Route is feasible.*"\
or a\
"*Arrived too late at customer*  \<Customer nr>. *Arrival time:*  \<Time of arrival>, *Time window:*  \<Customers time-window>"

``` bash
python .\prob3\2.prob3_kode\routes-feasibility_test.py
```