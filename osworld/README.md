# OSWorld

## Setup

1. Clone the original OSWorld repository:

```bash
git clone https://github.com/xlang-ai/OSWorld.git
```

2. Follow the configuration steps in the original repository to set up the environment and dependencies. We use [Docker](https://github.com/xlang-ai/OSWorld?tab=readme-ov-file#docker-server-with-kvm-support-for-the-better) to support the virtual machine.

3. For online training, we deploy the OSWorld environment on a dedicated machine. The environment interface is exposed as an API via [`env_api_wrapper.py`](./env_api_wrapper.py) for remote access by the agent. Multiple environment instances are managed in parallel using [`env_api_manager.py`](./env_api_manager.py). To launch the environment instances, simply place these two files in the OSWorld directory and run:

```bash
python env_api_manager.py
```

## Task Generation

### Initial Screenshot Generation

To generate your first screenshot for task generation:

1. Run OSWorld with any agent of your choice. If you don't want to execute any agent actions, you can set the agent's step count to 0.
2. This will create the initial screenshot needed for task generation.

### Task Generation Configuration

Before running the task generation script, you need to configure the following parameters in `task_generation.py`:

1. OpenAI API Key
2. Result directory path (location of screenshots, e.g., `./result/agent/pyautogui/screenshot/gpt-4o`)
3. Save directory path (default: `./generated_examples/examples`, must match `examples_dir` in `task_generation_meta.py`)
4. Option to generate infeasible tasks

In `task_generation_meta.py`, the `percentage` parameter determines the proportion of tasks that will be randomly selected for the final task set.

### Generating Tasks

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
