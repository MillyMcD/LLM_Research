#!/bin/bash

#activate the environment
source llm_env/bin/activate

#create notebook
jupyter notebook --ip 0.0.0.0 --port 9001 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''&