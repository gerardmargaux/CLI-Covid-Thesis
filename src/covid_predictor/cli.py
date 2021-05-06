import json
from pathlib import Path
import click
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

from covid_predictor.training import train_model
from covid_predictor.collection import actualize_trends, actualize_trends_using_daily, actualize_hospi
from covid_predictor.check_input import CheckCollection, CheckGlobalTraining, CheckTrainable, CheckUntrainable
from covid_predictor.utils import european_geocodes, french_region_and_be
from covid_predictor.prediction import prediction, plot_prediction, prediction_reference

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


@click.group('covid-predictor', help="Prediction tool of the number of hospitalizations based on open-data related to the coronavirus.")
def predictor():
    """Parent command for the group of commands."""
    pass


@predictor.command("collect", help="Collect the data needed for making a prediction.")
def run_collection():

    with open(Path(__file__).parents[2]/"config.json", 'r') as f:
        config = json.load(f)
    
    config = config["task"][0]
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
    

@predictor.command("run", help="Run the prediction of the number of hospitalizations.")
def run_training():

    with open(Path(__file__).parents[2]/"config.json", 'r') as f:
        config = json.load(f)

    config = config["task"][1]
    check = CheckGlobalTraining(**config)
    
    # Prediction for untrainable models
    if config["model_type"] == "untrainable":
        config_model = config["model"][1]
        check = CheckUntrainable(**config_model)
        final_pred, final_hospi = prediction_reference(model=config_model["name"], 
                                                 n_samples=config_model["days_to_use"], 
                                                 n_forecast=config_model["days_to_predict"], 
                                                 target=config["target"], 
                                                 geo=config["localisation"], 
                                                 europe=config["european_geo"])
    
    # Prediction for trainable models
    else:
        config_model = config["model"][0]
        check = CheckTrainable(**config_model)
        
        chosen_topics = {}
        for elem in config_model["topics"]:
            chosen_topics[elem] = list_topics[elem]
            
        """----------------------- TRAINING ----------------------- """
        # A pretained model already exists
        if os.path.exists(Path(__file__).parents[2]/f"models/{config_model['name']}_dg_param.json"):
            user_input = input("Do you want to train reuse a model that was already trained ? (y/n) ")
            
            # A previously trained model should be reused
            if user_input == "y":
                model = load_model(Path(__file__).parents[2]/f"models/{config_model['name']}_model.h5")
                with open(Path(__file__).parents[2]/f"models/{config_model['name']}_dg_param.json", 'r') as f:
                    parameters_dg = json.load(f)
            
            # A new model should be trained
            else:
                model, parameters_dg = train_model(type_model=config_model["name"], 
                                                epochs=config_model["epochs"], 
                                                n_samples=config_model["days_to_use"], 
                                                n_forecast=config_model["days_to_predict"], 
                                                target=config["target"],
                                                date_begin=config_model["date_begin"],
                                                list_topics=chosen_topics,
                                                cumsum=config_model["cumsum"],
                                                predict_one=config_model["predict_one"],
                                                europe=config["european_geo"],
                                                scaler_gen=config_model["scaler"])
        
                # Save the model trained
                model.save(Path(__file__).parents[2]/f"models/{config_model['name']}_model.h5")
        
                # Save the parameters of the data generator used
                with open(Path(__file__).parents[2]/f"models/{config_model['name']}_dg_param.json", "w") as f:
                    json.dump(parameters_dg, f, indent=4)
            
        # There is no trained model for this type of model         
        else:
            model, parameters_dg = train_model(type_model=config_model["name"], 
                                                epochs=config_model["epochs"], 
                                                n_samples=config_model["days_to_use"], 
                                                n_forecast=config_model["days_to_predict"], 
                                                target=config["target"],
                                                date_begin=config_model["date_begin"],
                                                list_topics=chosen_topics,
                                                cumsum=config_model["cumsum"],
                                                predict_one=config_model["predict_one"],
                                                europe=config["european_geo"],
                                                scaler_gen=config_model["scaler"])
        
            # Save the model trained
            model.save(Path(__file__).parents[2]/f"models/{config_model['name']}_model.h5")
    
            # Save the parameters of the data generator used
            with open(Path(__file__).parents[2]/f"models/{config_model['name']}_dg_param.json", "w") as f:
                json.dump(parameters_dg, f, indent=4)
    
        """----------------------- PREDICTION ----------------------- """
        loaded_df = pickle.load(open( Path(__file__).parents[2]/f"models/{config_model['name']}_merged_df.p", "rb" ))
        final_pred, final_hospi = prediction(model_name=config_model['name'],
                                    model=model,
                                    df=loaded_df,
                                    parameters_dg=parameters_dg,
                                    geo=config["localisation"])
        
    if config["print_results_on_terminal"]:
        print(f"{config_model['days_to_predict']}-days prediction of {config['target']} for {config['localisation']}")
        print(final_pred)
        
    if config["plot"]:
        plot_prediction(final_pred, final_hospi, geo=config["localisation"])
    

if __name__=="__main__":
    predictor()