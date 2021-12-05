import pandas as pd
import numpy as np
import json

from klarna.default_estimator import config


def get_member_ids():
    df = pd.read_csv('data/member_rfm_balance_name.csv')
    return list(df.flyer_id)

def get_default_score(id):
    df = pd.read_csv(config.output_data_path)
    line = df[df['uuid'] == id]
    # return member.to_json()
    return str(line.pd.values[0])
