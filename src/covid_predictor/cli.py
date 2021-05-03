import json
from pathlib import Path
import click

from covid_predictor.predictor import run_predictor
from covid_predictor.collector import actualize_trends, actualize_trends_using_daily, actualize_hospi
from covid_predictor.CheckPrediction import CheckPrediction
from covid_predictor.CheckCollection import CheckCollection
from covid_predictor.utils import european_geocodes, french_region_and_be

list_topics = {
        'Fièvre': '/m/0cjf0',
        'Mal de gorge': '/m/0b76bty',
        'Dyspnée': '/m/01cdt5',
        'Agueusie': '/m/05sfr2',
        'Anosmie': '/m/0m7pl',
        'Virus': '/m/0g9pc',
        'Épidémie': '/m/0hn9s',
        'Symptôme': '/m/01b_06',
        'Thermomètre': '/m/07mf1',
        'Grippe espagnole': '/m/01c751',
        'Paracétamol': '/m/0lbt3',
        'Respiration': '/m/02gy9_',
        'Toux': '/m/01b_21'
    }


@click.group('covid-predictor', help="Prediction model for covid-thesis.")
def predictor():
    """Parent command for the group of commands."""
    pass


@predictor.command("predict", help="Run the prediction model for a config.json file.")
def run_prediction():

    with open(Path(__file__).parents[2]/"config.json", 'r') as f:
        config = json.load(f)

    check = CheckPrediction(**config["task"][0])


@predictor.command("collect", help="Collect the data needed for making a prediction.")
def run_collection():

    with open(Path(__file__).parents[2]/"config.json", 'r') as f:
        config = json.load(f)
    
    config = config["task"][1]
    check = CheckCollection(**config)
    
    chosen_topics = {}
    for elem in config["topics"]:
        chosen_topics[elem] = list_topics[elem]

    # Actualize hospitalizations data
    actualize_hospi()
    
    # French regions and Belgium
    if not config["european_geo"]:  
        if config["collection"] == "hourly": # Hourly data collection
            actualize_trends(french_region_and_be, chosen_topics, plot=False, only_hourly=False, refresh_daily=True)
        else: # Daily data collection
            actualize_trends_using_daily(french_region_and_be, chosen_topics, plot=False, refresh=True)
    
    # European data
    else:
        if config["collection"] == "hourly": # Hourly data collection
            actualize_trends(european_geocodes, chosen_topics, plot=False, only_hourly=False, refresh_daily=True)
        else: # Daily data collection
            actualize_trends_using_daily(european_geocodes, chosen_topics, plot=False, refresh=True)
            
    print("End data collection")

if __name__=="__main__":
    predictor()