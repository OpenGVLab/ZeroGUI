from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional, Callable, Any


def preprocess_data(
    data: dict[str, Any],
    input_template: Optional[str] = None,
    input_key: str = "input",
    apply_chat_template: Optional[Callable[..., str]] = None
) -> str:
    """Preprocesses input data to generate a formatted prompt string.

    Processes input data differently based on whether a chat template function is provided:
    - When using chat templates: Converts chat data into a structured prompt string
    - Without chat templates: Applies simple string formatting using an optional template

    Args:
        data: Dictionary containing input data. Must contain the specified input_key.
        input_template: Optional template string for formatting (e.g., "Input: {}").
            Uses `{}` placeholder for the input content.
        input_key: Key in `data` containing the input content. Defaults to "input".
        apply_chat_template: Optional function that converts chat history to a prompt.
            Expected to accept chat messages, tokenize flag, and generation prompt flag.

    Returns:
        Formatted prompt string ready for model input.

    Examples:
        >>> data = {"input": "Hello world"}
        >>> preprocess_data(data, "User query: {}")
        'User query: Hello world'

        >>> data = {"input": [{"role": "user", "content": "Hi"}]}
        >>> preprocess_data(data, apply_chat_template=lambda x, **_: "Formatted: " + x[0]['content'])
        'Formatted: Hi'

        >>> data = {"input": "Short message"}
        >>> preprocess_data(data, apply_chat_template=lambda x, **_: "Chat: " + x[0]['content'])
        'Chat: Short message'
    """
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        prompt_max_len:int=-1,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)

            if prompt_max_len > 0:
                prompt_token_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
                if len(prompt_token_ids) > prompt_max_len:
                    continue 

            self.prompts.append(prompt)
        print(f"Dataset Length: {len(self.prompts)}")

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        # print('prompt dataset:', prompt)
        prompt = prompt.replace(
            "<|im_start|>user\n",
            "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
        )
        ## TODO: 兼容 qwen 的 trick 写法
        data_item = {'prompt': prompt, 'id': f'still3_rl_data-{idx}'}
        return data_item
        # return prompt
