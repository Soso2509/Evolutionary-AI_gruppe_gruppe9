# Evolutionary AI Portfolio - Group 9
## Prerequisites
Before you start working on the project, make sure you have the following installed
-  [Python 3.8+](https://www.python.org/downloads/)
- Your preferred code editor, like [Visual Studio Code](https://code.visualstudio.com/)
- [Git](https://git-scm.com/downloads) for version control

### Dependencies needed to run code
Found in "requirements.txt" file\
Spesific list found in each [problems README](#running-code-for-the-different-tasks-in-the-portfolio)


## Overview
- [Cloning the project](#cloning-the-code)
- [Setup Virtual Environment](#setup-of-virtual-environment)
    - [Activate Environment](#step-2-activating-the-environment)
    - [Install Dependencies](#step-3-install-dependencies)
- [Running code for different problems](#running-code-for-the-different-tasks-in-the-portfolio)
    - [Problem 2](./prob2/prob2_README.md)
    - [Problem 3](./prob3/prob3_README.md)
    - [Problem 4](./prob4/prob4_README.md)
- [Update dependencies](#updating-the-dependencies)
- [Deactivate Environment](#deactivating-the-environment)

## Cloning the code
### Step 1: Clone the project
Commands to do it through the terminal

SSH
``` bash
git clone git@github.com:Soso2509/Evolutionary-AI_gruppe_gruppe9.git
```
HTTPS
```bash
git clone https://github.com/Soso2509/Evolutionary-AI_gruppe_gruppe9.git
```

### Step 2: Navigate into the directory
```bash
cd Evolutionary-AI_gruppe_gruppe9
```
##  Setup of virtual environment
To isolate the projects dependencies, use a virtual environment
### Step 1: Create a virtual environment
**Windows**
```bash
python -m venv env
```

**MacOS/Linux**
```bash
python3 -m venv env
```

### Step 2: Activating the environment
**Windows**
```bash
.\env\Scripts\activate
```

**MacOS/Linux**
```bash
source env/bin/activate
```

### Step 3: Install dependencies
After the environment is active, install the dependencies
```bash
pip install -r requirements.txt
```

## Running code for the different tasks in the portfolio
When the environment is up and running, the code can be run\
Each problem has its own Markdown file with the terminal commands
- [Problem 2](./prob2/prob2_README.md)
- [Problem 3](./prob3/prob3_README.md)
- [Problem 4](./prob4/prob4_README.md)

## Updating the dependencies
*Make sure you are in the root of the project because that is where `requirements.txt` is located*\
Generate or update `requirements.txt` with packages that are installed in the current virtual environment
```bash
pip freeze > requirements.txt
```

Then commit and push `requirements.txt` to remote

## Deactivating the environment
After you are done working, you can deactivate the virtual environment with
```bash
deactivate
```