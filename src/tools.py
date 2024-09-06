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
    # Slack API functions
    kathy_channel = 'https://hooks.slack.com/services/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'
    emalie_channel = 'https://hooks.slack.com/services/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' # - Need to reconnect by Emalie if needed here - https://api.slack.com/apps/A06293X8D35/incoming-webhooks?
    file_channel = 'https://hooks.slack.com/services/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'

    if channel == 'kgarci18':
        url = kathy_channel
    elif channel == 'emcmaho7':
        url = emalie_channel
    else:
        raise "Channel is not recognised!"

    response = None
    if attachment:
        token = 'xoxb-508184378419-6100222024048-YpvTnypfCqlSlESxWeCs1eIn'
        client = WebClient(token)
        response = client.files_upload(channels='SIfMRI-modelling-alerts', title=attachment, file=attachment, initial_comment=msg)
    elif msg:
        webhook = WebhookClient(url)
        response = webhook.send(text=msg)
    return response

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