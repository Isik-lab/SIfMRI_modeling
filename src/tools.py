#!/usr/bin/env python
# coding: utf-8
import PIL
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import squareform
import math
from scipy.stats import spearmanr
import time
import torch
from slack_sdk.webhook import WebhookClient
from slack_sdk import WebClient
import sys
import os
import dotenv

def moving_grouped_average(outputs, skip=5, input_dim=0):
    from math import ceil as roundup # for rounding upwards
    return torch.stack([outputs[i*skip:i*skip+skip].mean(dim=input_dim) 
                        for i in range(roundup(outputs.shape[input_dim] / skip))])


def get_nearest_multiple(a, b):
    # Find the nearest multiple of b to a
    nearest_multiple = round(a / b) * b
    if nearest_multiple % 2 != 0:
        if (nearest_multiple - a) < (a - (nearest_multiple - b)):
            nearest_multiple += b
        else:
            nearest_multiple -= b
            
    return nearest_multiple # integer space

def send_slack(msg='', channel=None, attachment=None):
    """
     Helper function to send slack message to a webhook
     Arguments:
         msg: (str) The message to send. Defaults to 'SIfMRI-modelling-alerts'
         channel: (str) The channel or user to send to. Defaults to ''
         attachment (str) Optionally included filepath to an attachment that you want to include. Defaults to None
     Returns:
         slack-sdk response
    """
    env = Env()
    # Slack API functions
    kathy_hook = env.getFlag('kathy_channel')
    emalie_hook = env.getFlag('file_channel')  # - Defaults to file_channel.
    file_hook = env.getFlag('file_channel')

    if channel == 'kgarci18':
        url = kathy_hook
    elif channel == 'emcmaho7':
        url = emalie_hook
    else:
        url = file_hook

    response = None
    if attachment:
        token = 'xoxb-508184378419-6100222024048-YpvTnypfCqlSlESxWeCs1eIn'
        client = WebClient(token)
        response = client.files_upload(channels='SIfMRI-modelling-alerts', title=attachment, file=attachment, initial_comment=msg)
    elif msg:
        webhook = WebhookClient(url)
        response = webhook.send(text=msg)
    return response


class Env:
    def __init__(self):
        self.dotenv_file = dotenv.find_dotenv()
        self.dotenv = dotenv.load_dotenv(self.dotenv_file)
        print('Loaded env file.')

    def getFlag(self, name):
        flag = os.environ[str(name)]
        return flag

    def setFlag(self, name, value):
        name = str(name)
        value = str(value)
        os.environ[name] = value
        dotenv.set_key(self.dotenv_file, name, os.environ[name])
        return True


class TimeBlock:
    def __init__(self):
        self.start_time = None,
        self.end_time = None,
        self.elapsed = None,
        self.formatted_elapsed = None
    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def elapse(self, formatted=True):
        self.end()
        self.elapsed = self.end_time - self.start_time
        self.formatted_elapsed = time.strftime("%H:%M:%S", time.gmtime(self.elapsed))
        if formatted:
            return self.formatted_elapsed
        else:
            return self.elapsed