from .aguvis import AguvisAgent
from .uitars import UITARSAgent


def create_agent(args):
    if args.agent_type == 'aguvis':
        agent = AguvisAgent(
            tokenizer_path=args.pretrain,
            history_n=args.num_history,
            screen_size=(args.screen_width, args.screen_height),
        )
    elif args.agent_type == 'uitars':
        agent = UITARSAgent(
            tokenizer_path=args.pretrain,
            max_trajectory_length=args.agent_max_steps,
            history_n=args.num_history,
            screen_size=(args.screen_width, args.screen_height),
            action_space=args.agent_action_space,
            language=args.agent_prompt_language,
        )
    else:
        raise NotImplementedError
    return agent