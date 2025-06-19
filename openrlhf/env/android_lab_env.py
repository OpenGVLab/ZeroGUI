import time
import requests
import base64


def request_api_wrapper(url, data=None, try_max_times=5, method="POST", timeout=360):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            # response = requests.post(url=url, json=data, headers=headers)
            response = requests.request(method=method, url=url, json=data, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            is_success = response.get("success")
            if not is_success:
                message = response.get("message", None)
                print(f"API excecution error: {message}")
            else:
                return response
        except requests.RequestException as e:
            print(f"Request error, please check: {e}")
        except Exception as e:
            print(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


class RemoteAndroidLabEnv:
    def __init__(
            self,
            base_url: str,
            env_port: int = None,
            manager_port: int = None,
            connect_max_try: int = 5,
            llm_evaluator = None,
            test_task_llm_eval: bool = False,
    ):
        self.base_url=base_url
        self.env_port=env_port
        self.manager_port = manager_port
        self.connect_max_try = connect_max_try
        self.llm_evaluator = llm_evaluator
        self.test_task_llm_eval = test_task_llm_eval

        # create env api by manager
        if env_port is None:
            assert manager_port is not None, "use manager to assign an api"
            self.use_api_manager = True
            create_url = f"{base_url}:{manager_port}/create_env_api"
            response = request_api_wrapper(create_url, try_max_times=self.connect_max_try)
            self.env_id = response["env_id"]
            self.env_port = response["port"]
            time.sleep(5)
            print("creating env api done")
        else:
            self.use_api_manager = False

        # create env
        data = {}
        start_url = f"{base_url}:{self.env_port}/start"
        response = request_api_wrapper(start_url, data, try_max_times=self.connect_max_try)
        print("create env done")

    def get_task_config(self, domain, example_id, config_base_dir=None):
        data = {
            "task_id": domain,
            "example_id": example_id,
        }

        if config_base_dir is not None:
            data.update({"config_base_dir": config_base_dir})
        task_config_url = f"{self.base_url}:{self.env_port}/get_task_config"
        response = request_api_wrapper(task_config_url, data, try_max_times=self.connect_max_try)
        return response["task_config"]

    def reset(self, task_config):
        print("resetting env...")
        task_id = task_config["task_id"]
        reset_url = f"{self.base_url}:{self.env_port}/reset"
        response = request_api_wrapper(reset_url, {"task_id": task_id}, try_max_times=10)
        obs = response["obs"]
        obs["screenshot"] = base64.b64decode(obs["screenshot"]) # base64 -> bytes
        print("resetting env done.")
        return obs

    @property
    def vm_screen_size(self):
        vm_screen_size_url = f"{self.base_url}:{self.env_port}/vm_screen_size"
        response = request_api_wrapper(vm_screen_size_url, method='GET', try_max_times=self.connect_max_try)
        return response["vm_screen_size"]

    def step(self, action, pause=2):
        data = {
            "action": action,
            "pause": pause
        }
        step_url = f"{self.base_url}:{self.env_port}/step"
        # TODO: deal with unexpected step error, e.g., VM is closed by the action
        try:
            response = request_api_wrapper(step_url, data, try_max_times=self.connect_max_try)
            obs = response["obs"]
            obs["screenshot"] = base64.b64decode(obs["screenshot"]) # base64 -> bytes
            return obs, response["reward"], response["done"], response["info"]
        except Exception as e:
            print("Step failed, return None.")
            return None, -1, True, None

    def evaluate(self, task_config=None, trajectory=None):
        evaluate_url = f"{self.base_url}:{self.env_port}/evaluate"

        if self.test_task_llm_eval:
            try:
                outputs = self.llm_evaluator.evaluate_task(task_config, trajectory)
                return outputs
            except Exception as e:
                print("LLM Evaluation failed, return -1.")
                return {"reward": -1}
        # TODO: deal with unhandled eval error
        try:
            response = request_api_wrapper(evaluate_url, method='GET', try_max_times=self.connect_max_try)
            return {"reward": response["metric"]}
        except Exception as e:
            print("Evaluation failed, return -1.")
            return {"reward": -1}

    def close(self):
        # stop VM
        close_url = f"{self.base_url}:{self.env_port}/close"
        _ = request_api_wrapper(close_url, try_max_times=self.connect_max_try)
        # terminate env api
        if self.use_api_manager:
            terminate_url = f"{self.base_url}:{self.manager_port}/terminate_env_api"
            data = {"env_id": self.env_id}
            _ = request_api_wrapper(terminate_url, data, try_max_times=self.connect_max_try)
