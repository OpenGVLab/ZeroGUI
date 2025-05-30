# OSWorld Task Generation

This directory is dedicated to generating tasks in an OSWorld-like format. It provides tools and scripts for creating automated desktop interaction tasks using screenshots and natural language descriptions.

## Setup Instructions

1. Clone the original OSWorld repository:
```bash
git clone https://github.com/xlang-ai/OSWorld.git
```

2. Follow the configuration steps in the original repository to set up the environment and dependencies.

## Initial Screenshot Generation

To generate your first screenshot for task generation:

1. Run OSWorld with any agent of your choice. If you don't want to execute any agent actions, you can set the agent's step count to 0.
2. This will create the initial screenshot needed for task generation.

## Task Generation Configuration

Before running the task generation script, you need to configure the following parameters in `task_generation.py`:

1. OpenAI API Key
2. Result directory path (location of screenshots, e.g., `./result/agent/pyautogui/screenshot/gpt-4o`)
3. Save directory path (default: `./generated_examples/examples`, must match `examples_dir` in `task_generation_meta.py`)
4. Option to generate infeasible tasks

In `task_generation_meta.py`, the `percentage` parameter determines the proportion of tasks that will be randomly selected for the final task set.

## Generating Tasks

Once everything is configured:

1. Run the task generation script:
```bash
python task_generation.py
```

This will process your screenshots and generate task descriptions in the specified output directory.

2. Run the task metadata generation script:
```bash
python task_generation_meta.py
```

The generated task JSON file will be saved as `generated_examples/test_all.json`, which can be used for training or testing in OSWorld.

## Notes

- Before running any task generation scripts, create and validate all necessary destination directories and files
- Double check that your screenshot directory contains properly formatted and valid image files
