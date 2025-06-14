import ast
import re
from io import BytesIO
from typing import Dict
import math

import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor


SCREEN_LOGIC_SIZE = (1920, 1080)

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

UITARS_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
"""

UITARS_MOBILE_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
"""

UITARS_CALL_USR_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
"""

UITARS_USR_PROMPT_NOTHOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Action: ...
```
## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
## User Instruction
{instruction}
"""

UITARS_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None
    
def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)

def fix_click_output(output: str) -> str:
    # 直接匹配两个逗号分隔的数字，不考虑括号
    matches = re.findall(r'(\d+)\s*,\s*(\d+)', output)

    if matches:
        # 取最后一个匹配到的坐标
        x, y = matches[-1]
        return f"click(start_box='({x},{y})')"
    else:
        return None  # 没有找到任何有效的坐标时返回

def fix_drag_output(output: str) -> str:
    # 直接匹配两个逗号分隔的数字，不考虑括号
    matches = re.findall(r'(\d+)\s*,\s*(\d+)', output)

    if matches and len(matches) >= 2:
        # 取最后一个匹配到的坐标
        x1, y1 = matches[-2]
        x2, y2 = matches[-1]
        return f"drag(start_box='({x1},{y1})', end_box='({x2},{y2})')"
    else:
        return None  # 没有找到任何有效的坐标时返回

def parse_action_qwen2vl(text, factor, image_height, image_width):
    text = text.strip()
    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        elif "click(start_box" in action_str:
            # - Failed to parse action 'click(start_box='='x(409,173)')'
            # - Failed to parse action 'click(start_box='='\nstart_box='(430,348)')'
            # - Failed to parse action 'click(start_box='='\n' Data')'
            # - Failed to parse action 'click(start_box='237,72)'
            # - Failed to parse action 'click(start_box='='(492,348)')'
            # - Failed to parse action 'click(start_box='='\n(493,350)')'
            action_str_fixed = fix_click_output(action_str)
            if (action_str_fixed is not None) and (action_str_fixed != action_str):
                print('[CLICK ACTION FIXED]', action_str, '->', action_str_fixed)
                action_str = action_str_fixed
        elif "drag(start_box" in action_str:
            action_str_fixed = fix_drag_output(action_str)
            if (action_str_fixed is not None) and (action_str_fixed != action_str):
                print('[DRAG ACTION FIXED]', action_str, '->', action_str_fixed)
                action_str = action_str_fixed
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            continue
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                # if "[" in  ori_box:
                #     numbers = ori_box.replace("[", "").replace("]", "").split(",")
                # else:
                #     numbers = ori_box.replace("(", "").replace(")", "").split(",")
                numbers = ori_box.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")

                # Convert to float and scale by 1000
                float_numbers = [float(num) / factor for num in numbers]
                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions

def action_space_mapping(input_text: str) -> str:
    # 定义替换规则：正则表达式模式和对应的替换模板
    rules = [
        # 1. click(start_box='<|box_start|>(x1,y1)<|box_end|>')
        (
            r"click\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?'\)",
            lambda m: f'do(action="Tap", element=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}])'
        ),
        # 2. long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
        (
            r"long_press\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?', time=''?\)",
            lambda m: f'do(action="Long Press", element=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}])'
        ),
        # 2. long_press(start_box='<|box_start|>(x1,y1)<|box_end|>')
        (
            r"long_press\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?'\)",
            lambda m: f'do(action="Long Press", element=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}])'
        ),
        # 3. type(content='')
        (
            r"type\(content='((?:\'|[^'])*?)'\)",
            r'do(action="Type", text="\1")'
        ),
        # 4. scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
        (
            r"scroll\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?', end_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?'\)",
            lambda m: f'do(action="Swipe Precise", start=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}], end=[{int(m.group(3))/1000:.3f}, {int(m.group(4))/1000:.3f}])'
        ),
        # 5. scroll(direction='up')
        (
            r"scroll\(direction='((?:up|down|left|right))'\)",
            r'do(action="Swipe", direction="\1")'
        ),
        # 6. press_home()
        (
            r"press_home\(\)",
            r'do(action="Home")'
        ),
        # 7. press_back()
        (
            r"press_back\(\)",
            r'do(action="Back")'
        ),
        # 8. finished(content='')
        (
            r"finished\(content='((?:\'|[^'])*?)'\)",
            r'finish(message="\1")'
        ),
        # 9. finished()
        (
            r"finished\(\)",
            r'finish(message="")'
        ),
        # 10. drag(start_box='(624,470)', end_box='(288,505)')
        (
            r"drag\(start_box='\(([0-9]+),([0-9]+)\)', end_box='\(([0-9]+),([0-9]+)\)'\)",
            lambda m: f'do(action="Swipe Precise", start=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}], end=[{int(m.group(3))/1000:.3f}, {int(m.group(4))/1000:.3f}])'
        ),
        # 11. scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
        (
            r"scroll\(start_box='(?:<\|box_start\|>)?\(([0-9]+),([0-9]+)\)(?:<\|box_end\|>)?', direction='(down|up|left|right)'\)",
            lambda m: f'do(action="Swipe", element=[{int(m.group(1))/1000:.3f}, {int(m.group(2))/1000:.3f}], direction="{m.group(3)}")'
        ),
        # 12. open_app(app_name=\'\')
        (
            r"open_app\(app_name='([^']+)'\)",
            lambda m: f'do(action="Launch", app="{m.group(1)}")'
        )
    ]

    # 匹配整体输出格式：Thought: ...\nAction: ...\n
    output_pattern = r'(Thought:.*Action:.*)'
    
    def replace_action(match):
        line = match.group(1)
        # 提取 Action 部分
        action_match = re.search(r'Action: (.*?)(?=\n|$)', line)
        if not action_match:
            return line

        action = action_match.group(1)
        # 尝试每条替换规则
        for pattern, replacement in rules:
            if re.match(pattern, action):
                if callable(replacement):
                    # 使用lambda函数处理替换
                    action = re.sub(pattern, replacement, action)
                else:
                    # 普通替换
                    action = re.sub(pattern, replacement, action)
                break
        return action
        
    # 处理整个输入文本
    result = re.sub(output_pattern, replace_action, input_text, flags=re.DOTALL)
    return result

def parsing_response_to_android_action_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
    if isinstance(responses, dict):
        responses = [responses]
    action_code = ""
    for response_id, response in enumerate(responses):
        input_text = response["text"]
        action_code += action_space_mapping(input_text)

    return action_code


def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""

        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(3)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})

        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"

        elif action_type == "press":
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"
            
            elif hotkey == "space":
                hotkey = " "
                
            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"
            
        elif action_type == "keyup":
            key_to_up = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"
        
        elif action_type == "keydown":
            key_to_down = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"


        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)

                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")

            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code

def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


class UITARSAgent:
    def __init__(self,
                 tokenizer_path,
                 max_trajectory_length=15,
                 history_n=5,
                 screen_size=SCREEN_LOGIC_SIZE,
                 action_space='computer',
                 infer_mode='qwen2vl_user',
                 prompt_style='qwen2vl_user',
                 input_swap=False,
                 language='Chinese',
                 ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        self.processor = AutoProcessor.from_pretrained(tokenizer_path)
        self.max_trajectory_length = max_trajectory_length
        self.history_n = history_n
        self.screen_size = screen_size
        self.action_space = action_space
        self.infer_mode = infer_mode 
        self.prompt_style = prompt_style
        self.input_swap = input_swap
        self.language = language
        
        self.prompt_action_space = UITARS_ACTION_SPACE
        self.customize_action_parser = parse_action_qwen2vl
        self.action_parse_res_factor = 1000
        if self.infer_mode == "qwen2vl_user":
            self.prompt_action_space = UITARS_CALL_USR_ACTION_SPACE
        if action_space == 'mobile':
            self.prompt_action_space = UITARS_MOBILE_ACTION_SPACE
            self.action_code_mapper = parsing_response_to_android_action_code
        else:
            self.action_code_mapper = parsing_response_to_pyautogui_code
        
        self.prompt_template = UITARS_USR_PROMPT_THOUGHT
        
        if self.prompt_style == "qwen2vl_user":
            self.prompt_template = UITARS_USR_PROMPT_THOUGHT
        elif self.prompt_style == "qwen2vl_no_thought":
            self.prompt_template = UITARS_USR_PROMPT_NOTHOUGHT

        self.reset()

    def get_model_inputs(self, instruction: str, obs: Dict):
        
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts), \
            "The number of observations and actions should be the same."

        self.history_images.append(obs["screenshot"])

        base64_image = obs["screenshot"]
        self.observations.append(
            {"screenshot": base64_image, "accessibility_tree": None}
        )
        
        if self.infer_mode == "qwen2vl_user":
            user_prompt = self.prompt_template.format(
                instruction=instruction,
                action_space=self.prompt_action_space,
                language=self.language
            )
        elif self.infer_mode == "qwen2vl_no_thought":
            user_prompt = self.prompt_template.format(
                instruction=instruction
            )

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        max_pixels = 1350 * 28 * 28
        min_pixels = 100 * 28 * 28
        messages, images = [], []
        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        elif isinstance(self.history_images, list):
            pass
        else:
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")
        max_image_nums_under_32k = int(32768*0.75/max_pixels*28*28)
        if len(self.history_images) > max_image_nums_under_32k:
            num_of_images = min(5, len(self.history_images))
            max_pixels = int(32768*0.75) // num_of_images

        for turn, image in enumerate(self.history_images):
            # if len(images) >= 5:
            #     break
            try:
                image = Image.open(BytesIO(image))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")

            if image.width * image.height > max_pixels:
                """
                如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
                """
                resize_factor = math.sqrt(max_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                image = image.resize((width, height))
            if image.width * image.height < min_pixels:
                resize_factor = math.sqrt(min_pixels / (image.width * image.height))
                width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
                image = image.resize((width, height))

            if image.mode != "RGB":
                image = image.convert("RGB")

            images.append(image)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
        
        image_num = 0
        image_input_list = []
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # send at most history_n images to the model
                if history_idx + self.history_n > len(self.history_responses):
                    cur_image = images[image_num]
                    image_input_list.append(cur_image)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image", "image": ""}]
                    })
                    image_num += 1
                    
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": add_box_token(history_response)}]
                })

            cur_image = images[image_num]
            image_input_list.append(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image", "image": ""}]
            })
            image_num += 1
        else:
            cur_image = images[image_num]
            image_input_list.append(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image", "image": ""}]
            })
            image_num += 1

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = {"prompt": prompt_text, "multi_modal_data": {'image': image_input_list}}
        return inputs

    def parse_action(self, response: str):
        
        self.history_responses.append(response)
        self.thoughts.append(response)

        try:
            parsed_responses = self.customize_action_parser(
                response,
                self.action_parse_res_factor,
                self.screen_size[1],
                self.screen_size[0]
            )
        except Exception as e:
            print(f"Parsing action error: {response}, with error:\n{e}")
            return ["DONE"]

        actions = []
        for parsed_response in parsed_responses:
            if "action_type" in parsed_response:

                if self.action_space != 'mobile' and parsed_response["action_type"] == FINISH_WORD:
                    self.actions.append(actions)

                    return ["DONE"]
                
                elif parsed_response["action_type"] == WAIT_WORD:
                    self.actions.append(actions)
                    return ["WAIT"]
                
                elif parsed_response["action_type"] == ENV_FAIL_WORD:
                    self.actions.append(actions)
                    return ["FAIL"]

                elif parsed_response["action_type"] == CALL_USER:
                    self.actions.append(actions)
                    return ["FAIL"]

            try:
                pyautogui_code = self.action_code_mapper(
                    parsed_response,
                    self.screen_size[1],
                    self.screen_size[0],
                    self.input_swap
                )
                actions.append(pyautogui_code)
            except Exception as e:
                print(f"Parsing pyautogui code error: {parsed_response}, with error:\n{e}")

        self.actions.append(actions)

        if len(self.history_responses) >= self.max_trajectory_length:
            # Default to FAIL if exceed max steps
            actions = ["FAIL"]

        return actions

    def reset(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
