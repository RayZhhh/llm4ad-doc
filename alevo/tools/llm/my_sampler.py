from __future__ import annotations

import json
import random
import time
from typing import Optional, List, Dict

import numpy as np
import requests

from alevo.base import Sampler


class MyDispatch(Sampler):

    def __init__(self):
        super().__init__()

    def draw_sample(self, prompt: Optional[str, List[Dict[str, str]]], *args, **kwargs) -> str:
        while True:
            try:
                random_url = f'http://127.0.0.1:{12000 + np.random.randint(1, 17)}/completions',
                # print(f'rand_url: {random_url}')
                return self._do_request(prompt, random_url)
            except Exception as e:
                # print(e)
                continue

    def _do_request(self, content: str, url) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference (inorder to decease the sample delay)
        data = {
            'url': url,
            'prompt': content,
            'repeat_prompt': 1,
            'params': {
                'max_new_tokens': 4096,
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            'http://172.18.36.43:16666/dispatch',
            data=json.dumps(data),
            headers=headers
        )
        if response.status_code == 200:
            response = response.json()['content']
            return response[0]


class MySampler(Sampler):

    def __init__(self):
        super().__init__()

    def draw_sample(self, prompt: Optional[str, List[Dict[str, str]]], *args, **kwargs) -> str:
        prompt = prompt.strip()
        random_url = f'http://127.0.0.1:{12000 + np.random.randint(1, 32)}/completions',
        return _request(prompt, random_url)


def _request(prompt, url):
    while True:
        try:
            data = {
                'prompt': prompt,
                'repeat_prompt': 1,
                'system_prompt': '',
                'stream': False,
                'params': {
                    'max_new_tokens': 4096,
                    'temperature': None,
                    'top_k': None,
                    'top_p': None,
                    'add_special_tokens': False,
                    'skip_special_tokens': True,
                }
            }

            headers = {'Content-Type': 'application/json'}
            record_time = time.time()
            response = requests.post(url, data=json.dumps(data), headers=headers)
            durations = time.time() - record_time
            print(f'time: {durations}s')

            if response.status_code == 200:
                print(f'Query time: {durations}')
                # print(f'Response: {response.json()}')
                content = response.json()["content"][0]
                return content
        except Exception as e:
            print(e)
            continue
