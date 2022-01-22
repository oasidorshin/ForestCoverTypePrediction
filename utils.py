import datetime
import random, os
import numpy as np


def get_timestamp():
    timestamp = datetime.datetime.now().isoformat(' ', 'minutes')  # without seconds
    timestamp = timestamp.replace(' ', '-').replace(':', '-')
    return timestamp

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)