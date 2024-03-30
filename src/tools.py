#!/usr/bin/env python
# coding: utf-8
import PIL
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import squareform
import math
from scipy.stats import spearmanr
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

def send_slack(msg='', channel='SIfMRI-modelling-alerts', attachment=None):
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
    emalie_channel = 'https://hooks.slack.com/services/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'
    file_channel = 'https://hooks.slack.com/services/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'

    if channel == 'kgarci18':
        url = kathy_channel
    elif channel == 'emcmaho7':
        url = emalie_channel
    else:
        url = file_channel

    response = None
    if attachment:
        token = 'xoxb-508184378419-6100222024048-YpvTnypfCqlSlESxWeCs1eIn'
        client = WebClient(token)
        response = client.files_upload(channels='SIfMRI-modelling-alerts', title=attachment, file=attachment, initial_comment=msg)
    elif msg:
        webhook = WebhookClient(url)
        response = webhook.send(text=msg)
    return response
