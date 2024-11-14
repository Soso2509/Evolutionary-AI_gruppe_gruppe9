# Running code from Problem 4
## Prerequisites to run the code
Following is required
- **Jupyter notebook** to read the `Taxi.ipynb` file
- Dependencies:\
(*If installed `requirements.txt` they are already installed*)
    - Gym
    - matplotlib
    - moviepy
    - pygame
    - numpy version 1.26.3
    - torch

## Potential problems
If the code does run not in the notebook, try **extracting them to individual python files**\
Running them with
```bash
python .\prob4\2.prob4_kode\<file_name>
```

If DDQN (*Double Deep Q-learning*) does not provide feasible results follow these steps:
1. Try again, sometimes the algorithm does not find a suitable solution within the 2000 episodes
2. Try restarting the IDE to flush the cache (pytorch)