"""
This script is used to select a certain percentage of tasks from all generated task JSON files.
A JSON file will be generated as the meta file for task set.
The meta file will be used for evaluation or RL training.
"""
import json
import random
import os
import math

# Define the percentage of tasks to select
percentage = 0.5

# Path to the directory containing all generated task JSON files
examples_dir = './generated_examples/examples'

# Get all JSON files in the examples directory and its subdirectories
all_tasks = {}
for category in os.listdir(examples_dir):
    category_path = os.path.join(examples_dir, category)
    if os.path.isdir(category_path):
        all_tasks[category] = []
        for filename in os.listdir(category_path):
            if filename.endswith('.json'):
                # Extract the task ID (assuming filename is taskid.json)
                task_id = os.path.splitext(filename)[0]
                all_tasks[category].append(task_id)

# Create a new dictionary to store the selected tasks
selected_data = {}

# For each category, randomly select the specified percentage of tasks
for category, task_ids in all_tasks.items():
    if task_ids:
        # Randomly select the tasks
        num_to_select = math.ceil(len(task_ids) * percentage)
        selected_tasks = random.sample(task_ids, num_to_select)
        selected_data[category] = selected_tasks

# Ensure the output directory exists
os.makedirs('generated_examples', exist_ok=True)

# Write the selected tasks to the new file
with open('generated_examples/test_all.json', 'w') as f:
    json.dump(selected_data, f, indent=1)

print(f"Successfully selected {percentage*100}% of tasks and saved to generated_examples/test_all.json")
