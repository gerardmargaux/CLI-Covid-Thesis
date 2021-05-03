import json
from pathlib import Path
import pandas as pd

from covid_predictor.utils import *


def run_predictor(name, moment):    
    hello_string = hello_printer(name=name)
    goodbye_string = sub_functionA(moment=moment, name=name)

    result = {
        "hello": hello_string,
        "goodbye": goodbye_string
        }

    return result
