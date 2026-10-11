"""Dependency-free client for the local code-generation server."""

import json
from urllib.request import Request, urlopen


class LocalModel:
    def __init__(self, url="http://127.0.0.1:8123", *, timeout=180):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self):
        with urlopen(self.url + "/health", timeout=5) as response:
            return json.load(response)

    def generate(self, messages, *, seed=0, temperature=0.2, max_tokens=2048):
        request = Request(
            self.url + "/generate",
            data=json.dumps(
                dict(
                    messages=messages,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)
