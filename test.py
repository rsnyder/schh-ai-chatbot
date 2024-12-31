#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import requests

url = 'http://localhost:8088/stream'
message = 'Hello, how are you?'
data = {"content": message}

headers = {'Content-type': 'application/json'}

with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
  for chunk in r.iter_content(1024):
    print(chunk)