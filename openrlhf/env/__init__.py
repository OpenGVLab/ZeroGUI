from .osworld_env import RemoteDesktopEnv
from .android_lab_env import RemoteAndroidLabEnv
from .llm_eval import LLMEvaluator


def create_env(args, env_idx=0):
    if args.use_llm_evaluator:
        api_urls = args.api_base_url.split(',')
        api_url = api_urls[env_idx % len(api_urls)]
        llm_evluator = LLMEvaluator(
            api_type=args.api_type,
            model=args.api_model,
            base_url=api_url,
            api_key=args.api_key,
            prompt_file=args.eval_prompt_file,
            prompt_dir=args.eval_prompt_dir,
            temperature=args.llm_eval_temperature,
            voting_type=args.llm_eval_voting_type,
            voting_num=args.llm_eval_voting_num,
        )
    else:
        llm_evluator = None

    if args.env_type == 'osworld':
        # use multiple env urls
        urls = args.env_url.split(',')
        url = urls[env_idx % len(urls)]
        env = RemoteDesktopEnv(
            base_url=url,
            env_port=args.env_port,
            manager_port=args.env_manager_port,
            action_space=args.action_space,
            screen_size=(args.screen_width, args.screen_height),
            headless=True,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type
            in ["a11y_tree", "screenshot_a11y_tree", "som"],
            llm_evaluator=llm_evluator,
            test_task_llm_eval=args.test_task_llm_eval,
        )
        env_config = {
            'max_steps': args.agent_max_steps,
            'env_type': args.env_type
        }
    elif args.env_type == 'android_lab':
        urls = args.env_url.split(',')
        url = urls[env_idx % len(urls)]
        env = RemoteAndroidLabEnv(
            base_url=url,
            env_port=args.env_port,
            manager_port=args.env_manager_port,
            llm_evaluator=llm_evluator,
            test_task_llm_eval=args.test_task_llm_eval,
        )
        env_config = {
            'max_steps': args.agent_max_steps,
            'env_type': args.env_type
        }
    else:
        raise NotImplementedError
    return env, env_config