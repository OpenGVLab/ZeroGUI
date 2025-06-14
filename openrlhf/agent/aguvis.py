import re
from PIL import Image
from io import BytesIO
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer


AGUVIS_SYS_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."""

AGUVIS_PLANNING_PROMPT = """Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}.

Previous actions:
{previous_actions}
"""

AGUVIS_RECIPIENT_SELF_PLAN = "<|recipient|>"


SCREEN_LOGIC_SIZE = (1920, 1080)


def parse_aguvis_response(input_string, screen_logic_size=SCREEN_LOGIC_SIZE) -> Tuple[str, List[str]]:
    if input_string.lower().startswith("wait"):
        return "WAIT", "WAIT"
    elif input_string.lower().startswith("done"):
        return "DONE", "DONE"
    elif input_string.lower().startswith("fail"):
        return "FAIL", "FAIL"

    try:
        lines = input_string.strip().split("\n")
        lines = [line for line in lines if line.strip() != "" and line.strip() != "assistantall"]
        # low_level_instruction = lines[0]
        low_level_instruction = "None"
        for line in lines:
            if line.startswith("Action:"):
                low_level_instruction = line.strip("Action:").strip()

        pyautogui_index = -1

        for i, line in enumerate(lines):
            if line.strip() == "assistantos" or line.strip().startswith("pyautogui"):
                pyautogui_index = i
                break

        if pyautogui_index == -1:
            print(f"Error: Could not parse response {input_string}")
            return None, None

        pyautogui_code_relative_coordinates = "\n".join(lines[pyautogui_index:])
        pyautogui_code_relative_coordinates = pyautogui_code_relative_coordinates.replace("assistantos", "").strip()
        corrected_code = correct_pyautogui_arguments(pyautogui_code_relative_coordinates)

        parsed_action = _pyautogui_code_to_absolute_coordinates(corrected_code, screen_logic_size)
        return low_level_instruction, parsed_action
    except Exception as e:
        print(f"Error: Could not parse response {input_string}")
        return None, None

def correct_pyautogui_arguments(code: str) -> str:
    function_corrections = {
        'write': {
            'incorrect_args': ['text'],
            'correct_args': [],
            'keyword_arg': 'message'
        },
        'press': {
            'incorrect_args': ['key', 'button'],
            'correct_args': [],
            'keyword_arg': None
        },
        'hotkey': {
            'incorrect_args': ['key1', 'key2', 'keys'],
            'correct_args': [],
            'keyword_arg': None
        },
    }

    lines = code.strip().split('\n')
    corrected_lines = []

    for line in lines:
        line = line.strip()
        match = re.match(r'(pyautogui\.(\w+))\((.*)\)', line)
        if match:
            full_func_call = match.group(1)
            func_name = match.group(2)
            args_str = match.group(3)

            if func_name in function_corrections:
                func_info = function_corrections[func_name]
                args = split_args(args_str)
                corrected_args = []

                for arg in args:
                    arg = arg.strip()
                    kwarg_match = re.match(r'(\w+)\s*=\s*(.*)', arg)
                    if kwarg_match:
                        arg_name = kwarg_match.group(1)
                        arg_value = kwarg_match.group(2)

                        if arg_name in func_info['incorrect_args']:
                            if func_info['keyword_arg']:
                                corrected_args.append(f"{func_info['keyword_arg']}={arg_value}")
                            else:
                                corrected_args.append(arg_value)
                        else:
                            corrected_args.append(f'{arg_name}={arg_value}')
                    else:
                        corrected_args.append(arg)

                corrected_args_str = ', '.join(corrected_args)
                corrected_line = f'{full_func_call}({corrected_args_str})'
                corrected_lines.append(corrected_line)
            else:
                corrected_lines.append(line)
        else:
            corrected_lines.append(line)

    corrected_code = '\n'.join(corrected_lines)
    return corrected_code

def split_args(args_str: str) -> List[str]:
    args = []
    current_arg = ''
    within_string = False
    string_char = ''
    prev_char = ''
    for char in args_str:
        if char in ['"', "'"]:
            if not within_string:
                within_string = True
                string_char = char
            elif within_string and prev_char != '\\' and char == string_char:
                within_string = False
        if char == ',' and not within_string:
            args.append(current_arg)
            current_arg = ''
        else:
            current_arg += char
        prev_char = char
    if current_arg:
        args.append(current_arg)
    return args


def _pyautogui_code_to_absolute_coordinates(pyautogui_code_relative_coordinates, logical_screen_size=SCREEN_LOGIC_SIZE):
    """
    Convert the relative coordinates in the pyautogui code to absolute coordinates based on the logical screen size.
    """
    import re
    import ast

    width, height = logical_screen_size

    pattern = r'(pyautogui\.\w+\([^\)]*\))'

    matches = re.findall(pattern, pyautogui_code_relative_coordinates)

    new_code = pyautogui_code_relative_coordinates

    for full_call in matches:
        func_name_pattern = r'(pyautogui\.\w+)\((.*)\)'
        func_match = re.match(func_name_pattern, full_call, re.DOTALL)
        if not func_match:
            continue

        func_name = func_match.group(1)
        args_str = func_match.group(2)

        try:
            parsed = ast.parse(f"func({args_str})").body[0].value
            parsed_args = parsed.args
            parsed_keywords = parsed.keywords
        except SyntaxError:
            continue

        function_parameters = {
            'click': ['x', 'y', 'clicks', 'interval', 'button', 'duration', 'pause'],
            'moveTo': ['x', 'y', 'duration', 'tween', 'pause'],
            'moveRel': ['xOffset', 'yOffset', 'duration', 'tween', 'pause'],
            'dragTo': ['x', 'y', 'duration', 'button', 'mouseDownUp', 'pause'],
            'dragRel': ['xOffset', 'yOffset', 'duration', 'button', 'mouseDownUp', 'pause'],
            'doubleClick': ['x', 'y', 'interval', 'button', 'duration', 'pause'],
        }

        func_base_name = func_name.split('.')[-1]

        param_names = function_parameters.get(func_base_name, [])

        args = {}
        for idx, arg in enumerate(parsed_args):
            if idx < len(param_names):
                param_name = param_names[idx]
                arg_value = ast.literal_eval(arg)
                args[param_name] = arg_value

        for kw in parsed_keywords:
            param_name = kw.arg
            arg_value = ast.literal_eval(kw.value)
            args[param_name] = arg_value

        updated = False
        if 'x' in args:
            try:
                x_rel = float(args['x'])
                x_abs = int(round(x_rel * width))
                args['x'] = x_abs
                updated = True
            except ValueError:
                pass
        if 'y' in args:
            try:
                y_rel = float(args['y'])
                y_abs = int(round(y_rel * height))
                args['y'] = y_abs
                updated = True
            except ValueError:
                pass
        if 'xOffset' in args:
            try:
                x_rel = float(args['xOffset'])
                x_abs = int(round(x_rel * width))
                args['xOffset'] = x_abs
                updated = True
            except ValueError:
                pass
        if 'yOffset' in args:
            try:
                y_rel = float(args['yOffset'])
                y_abs = int(round(y_rel * height))
                args['yOffset'] = y_abs
                updated = True
            except ValueError:
                pass

        if updated:
            reconstructed_args = []
            for idx, param_name in enumerate(param_names):
                if param_name in args:
                    arg_value = args[param_name]
                    if isinstance(arg_value, str):
                        arg_repr = f"'{arg_value}'"
                    else:
                        arg_repr = str(arg_value)
                    reconstructed_args.append(arg_repr)
                else:
                    break

            used_params = set(param_names[:len(reconstructed_args)])
            for kw in parsed_keywords:
                if kw.arg not in used_params:
                    arg_value = args[kw.arg]
                    if isinstance(arg_value, str):
                        arg_repr = f"{kw.arg}='{arg_value}'"
                    else:
                        arg_repr = f"{kw.arg}={arg_value}"
                    reconstructed_args.append(arg_repr)

            new_args_str = ', '.join(reconstructed_args)
            new_full_call = f"{func_name}({new_args_str})"
            new_code = new_code.replace(full_call, new_full_call)

    return new_code


class AguvisAgent:
    def __init__(self,
                 tokenizer_path,
                 history_n=3,
                 screen_size=SCREEN_LOGIC_SIZE):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        self.history_n = history_n
        self.screen_size= screen_size

        self.observations = []
        self.actions = []
        self.thoughts = []

    def get_model_inputs(self, instruction: str, obs: Dict):
        _observations = self.observations[-self.history_n:]
        _actions = self.actions[-self.history_n:]
        _thoughts = self.thoughts[-self.history_n:]

        previous_actions = "\n".join([f"Step {i+1}: {action}" for i, action in enumerate(_actions)]) if _actions else "None"
        user_prompt = AGUVIS_PLANNING_PROMPT.format(
                        instruction=instruction,
                        previous_actions=previous_actions)

        messages = []
        messages.append({
            "role": "system",
            "content": [
                {"type": "text", "text": AGUVIS_SYS_PROMPT}
            ],
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": user_prompt},
            ],
        })
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text += AGUVIS_RECIPIENT_SELF_PLAN

        image = Image.open(BytesIO(obs["screenshot"]))
        inputs = {"prompt": prompt_text, "multi_modal_data": {'image': [image]}}
        return inputs

    def parse_action(self, response: str):
        low_level_instruction, pyautogui_actions = parse_aguvis_response(response, self.screen_size)
        self.thoughts.append(response)
        self.actions.append(low_level_instruction)
        actions = [pyautogui_actions] if pyautogui_actions is not None else ["DONE"]
        return actions

    def reset(self):
        self.observations = []
        self.actions = []
        self.thoughts = []
