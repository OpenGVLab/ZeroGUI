"""
This script is used to generate task instructions for the OSWorld benchmark.
Each task configuration will be saved to a JSON file.
"""
import json
import uuid
from typing import TypedDict, Optional
from openai import OpenAI
import os
import requests
import time
import logging
import base64
import random
from task_generation_prompts import prompts, refine_instruction_prompt, prompts_infeasible_tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('task_generation.log')  # Output to file
    ]
)
logger = logging.getLogger(__name__)

RESULT_DIR = './path-to-result-directory'    # Change this to the path to the result directory (where you have run the OSWorld benchmark)
YOUR_API_KEY = 'YOUR-API-KEY'                # Change this to your OpenAI API key
OUTPUT_DIR = './generated_examples/examples' # Change this to the path to the output directory (where you want to save the generated task configurations)
GENERATE_INFEASIBLE_TASKS = False           # Change this to True if you want to generate infeasible tasks

NUM_INSTRUCTIONS = 10                       # The number of instructions generating each time
ROUNDS_PER_DOMAIN = 100                     # The number of rounds generating for each domain each time
API_URL = "https://api.openai.com/v1/chat/completions"

class TaskConfig(TypedDict):
    id: str
    snapshot: str
    instruction: str
    source: str
    trajectory: str
    related_apps: list[str]
    evaluator: dict
    config: dict

class TaskConfigGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.system_prompt = """You are a task configuration generator for Ubuntu-based computer use tasks. 
        Generate practical, executable instructions that can be performed on an Ubuntu machine.
        The tasks should be clear, specific, and achievable using standard Ubuntu applications and browsers.
        Focus on common user interactions."""
        self.system_infeasible_prompt = """You are a task configuration generator for Ubuntu-based computer use tasks. 
        Generate infeasible instructions on an Ubuntu machine, as for training the computer use agent to recognize infeasible tasks.
        Focus on common user interactions."""
    
    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise

    def call_llm(self, prompt: str, retry_times: int = 5, retry_delay: int = 5) -> str:
        for attempt in range(retry_times):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "model": 'gpt-4o-2024-11-20',
                    "messages": prompt,
                    "max_tokens": 1000,
                }
            )
            if response.status_code != 200:
                logger.warning(f"API request failed: {response.text}")
                if attempt < retry_times - 1:  # Don't sleep on the last attempt
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return ""
            else:
                return response.json()['choices'][0]['message']['content']

    def _generate_instruction(self, domain: str, num_instructions: int, if_infeasible: bool = False) -> str:
        """Generate a task instruction using OpenAI."""
        try:
            logger.info("Generating task instruction")
            if if_infeasible:
                messages = [{"role": "system", "content": self.system_infeasible_prompt}]
            else:
                messages = [{"role": "system", "content": self.system_prompt}]
            
            # Use a random example setup from the evaluation examples
            original_task_dir = f'./evaluation_examples/examples/{domain}'
            original_task_files = [f for f in os.listdir(original_task_dir) if f.endswith('.json')]
            original_task_file = random.choice(original_task_files)
            with open(os.path.join(original_task_dir, original_task_file), 'r') as f:
                original_task = json.load(f)
            config = original_task['config'] if 'config' in original_task else []

            # Load the 0-th step screenshot, you should have run OSWorld benchmark before to generate the screenshot
            screenshot_dir = f'{RESULT_DIR}/{domain}/{original_task["id"]}/step_0.png'
            # If the screenshot does not exist, try another example
            while not os.path.exists(screenshot_dir):
                logger.info(f"Screenshot {screenshot_dir} does not exist, trying another example")
                original_task_file = random.choice(original_task_files)
                with open(os.path.join(original_task_dir, original_task_file), 'r') as f:
                    original_task = json.load(f)
                config = original_task['config'] if 'config' in original_task else []
                screenshot_dir = f'{RESULT_DIR}/{domain}/{original_task["id"]}/step_0.png'

            if if_infeasible:
                user_content = [{"type": "text", "text": prompts_infeasible_tasks[domain].format(num_instructions=num_instructions)}]
            else:
                user_content = [{"type": "text", "text": prompts[domain].format(num_instructions=num_instructions)}]
            
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._encode_image(screenshot_dir)}",
                    "detail": "high",
                }})

            messages.append({"role": "user", "content": user_content})
            response = self.call_llm(messages)
            logger.debug(f"Generated instruction: {response}")
            return response, config
        except Exception as e:
            logger.error(f"Error generating instruction: {str(e)}", exc_info=True)
            raise Exception(f"Error generating instruction: {str(e)}")

    def _refine_instruction(self, instruction: str) -> str:
        """Refine the instruction using OpenAI."""
        # Remove the index number from the instruction if it exists
        import re
        pattern = re.compile(r'^\d+\.\s')
        if pattern.match(instruction):
            instruction = pattern.sub('', instruction, 1)
        try:
            logger.info("Refining instruction")
            messages = [{"role": "system", "content": "You are a task configuration generator for Ubuntu-based computer use tasks."}]
            messages.append({"role": "user", "content": refine_instruction_prompt.format(instruction=instruction)})
            response = self.call_llm(messages)
            logger.debug(f"Refined instruction: {response}")
            return response
        except Exception as e:
            logger.error(f"Error refining instruction: {str(e)}", exc_info=True)
            raise Exception(f"Error refining instruction: {str(e)}")

    def generate_config(self, domain: str, 
                        num_instructions: int = 1, 
                        refine_instruction: bool = False,
                        if_infeasible: bool = False) -> TaskConfig:
        """Generate a complete task configuration."""
        logger.info("Generating task configuration")
        
        # Generate the instruction using OpenAI
        instructions, config = self._generate_instruction(domain, num_instructions, if_infeasible)
        # Create the task configuration
        instructions = instructions.split('\n')
        task_configs = []
        for instruction in instructions:
            if instruction == "":
                continue
            if refine_instruction:
                instruction = self._refine_instruction(instruction)
            # Generate a unique UUID for the task
            task_id = str(uuid.uuid4())
            logger.debug(f"Generated task ID: {task_id}")
            task_config: TaskConfig = {
                "id": task_id,
                "snapshot": domain,
                "instruction": instruction,
                "related_apps": [domain],
                "source": "LLM generated",
                "trajectory": "trajectories/",
                "evaluator": {"func": "LLM"} if not if_infeasible else {"func": "infeasible"},
                "config": config
            }
            task_configs.append(task_config)
        logger.info("Task configuration generated successfully")
        logger.debug(f"Configurations: {task_configs}")
        return task_configs
        

    def save_config(self, task_config: TaskConfig, output_dir: str = "configs") -> str:
        """Save the configuration to a JSON file."""
        try:
            logger.info(f"Saving configuration to directory: {output_dir}")
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with task ID
            filename = os.path.join(output_dir, f"{task_config['id']}.json")
            logger.debug(f"Saving to file: {filename}")
            
            # Save configuration to JSON file
            with open(filename, 'w') as f:
                json.dump(task_config, f, indent=2)
            
            logger.info("Configuration saved successfully")
            return filename
        except Exception as e:
            logger.error(f"Error saving task config: {str(e)}", exc_info=True)
            raise Exception(f"Error saving task config: {str(e)}")

def main():
    # Example usage
    try:
        logger.info("Starting task configuration generation")
        # Initialize generator
        generator = TaskConfigGenerator(api_key=YOUR_API_KEY)
        if_infeasible = GENERATE_INFEASIBLE_TASKS
        num_instructions = NUM_INSTRUCTIONS
        refine_instruction = True

        # Generate configuration
        for domain in ["chrome", "gimp", "libreoffice_calc", "libreoffice_impress", "libreoffice_writer", "thunderbird", "vlc", "vs_code", "multi_apps", "os"]:
            # generate ROUNDS_PER_DOMAIN rounds per domain
            for _ in range(ROUNDS_PER_DOMAIN):
                task_configs = generator.generate_config(domain=domain, num_instructions=num_instructions, refine_instruction=refine_instruction, if_infeasible=if_infeasible)
                for task_config in task_configs:
                    filename = generator.save_config(task_config, output_dir=f"{OUTPUT_DIR}/{domain}")
                    logger.info(f"Generated task configuration saved to: {filename}")
                    print(f"Generated task configuration saved to: {filename}")
                    print("Configuration contents:")
                    print(json.dumps(task_config, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 