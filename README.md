# Evolutionary AI Portfolio - Group 9
## Prerequisites
Before you start working on the project, make sure you have the following installed
-  [Python 3.8+](https://www.python.org/downloads/)
- Your preferred code editor, like [Visual Studio Code](https://code.visualstudio.com/)
- [Git](https://git-scm.com/downloads) for version control

## Overview
- []()

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

## Some other useful git stuff
### Process of making changes
1. `git status` _See if changes done locally_
2. `git pull` _Retrieving changes from remote_
3. _Make changes_
4. `git add <file1> <file2> ...` _adds file/changes to ones tracked by git_\
OR\
`git add .` _adds all un-tracked files/changes_
5. `git commit -m "message"` _commit all files and adds a message in terminal (can also be added in editor)_
6. `git push` _Pushes all changes committed til remote_

### Branches
`git branch <new-branch-name>` _Creates a new local branch_, Remember this branch is local and is not able to be seen or interacted with until you push it to remote

`git branch` _Shows list of branches, the one with a * is the branch you are currently "standing" in_

`git checkout <branch or commit>` _Can be uses to "switch" to previous commits or switch branches_

`git switch <branch>` _To switch between branches_

`git push -u origin <branch>` _Pushes branch to remote_

`git merge <branch>` _Merges changes from another branch into the branch you're in_