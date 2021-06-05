# CLI COVID-THESIS

## How to install it ? 

1. Install Miniconda

    Link : [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Create a virtual environment (in the terminal)
```
conda create --name thesis python=3.8 --yes
```

3. Activate the environment
```
conda activate thesis
```

3. Go to the folder
```
cd path/to/repository

```

4. Install the package
```
pip install -e .
```

5. Use it
```
covid-predictor predict
covid-predictor collect
```

## How to use it ? 

You will find several parameters that can be changed in the _config.json_ file before running the code. These parameters are the following : 

### Collection of data : 

* **name** : In this case, the name of the task can be _collection_ or _collect_.

* **collection** : Refers to the type of collection that should be done. Can be _hourly_ to collect Trends data on an hour-by-hour basis, _daily_ to collect data on a day-by-day basis or _minimal_ to do a minimal data collection.

* **topics** : List of topics that will be used to collect data. 

* **european_geo** : is _true_ if the european geocodes should be considered. Is _false_ if the localisation that should be considered is Belgium + some French regions.

### Prediction of the number of hospitalizations :

To make a prediction of the number of hospitalizations in a certain region, several global parameters must be defined.

* **model_type** : define the type of model that will be used for the prediction. Can be _trainable_ or _untrainable_.

* **target** : This describes the target that the model will try to predict. This can be the total number of hospitalisations (_TOT_HOSP_) or the number of new hospitalisations (_NEW_HOSP_).

* **localisation** : This parameter defines the localisation for which the prediction should be made. For example, if we want to predict the number of hospitalizations in Belgium, the localisation will be "BE".

* **european_geo** :  If this parameter is _true_ then the model will be trained using data relative to the european countries. On the contrary, if it is _false_, the model will be trained data relative to Belgium and French regions.

* **print_results_on_terminal** : This parameter can be set to _true_ in order to print the results of the prediction on the terminal when the execution is terminated.

* **plot** : This parameter can be set to _true_in order to generate an interactive plot representing the prediction that was done for the parameters that were chosen by the user.

### Prediction of the number of hospitalizations with a trainable model : 

* **name** : This represents the name of the trainable model that should be trained. It can be _assembler_, _dense_, _encoder-decoder_ or _simple-auto-encoder_.

* **days_to_use** :  This represents the number of days that should be used by the model to make the prediction.

* **days_to_predict** :  This parameter is used to determine the number of days that should be predicted by the model. We recommend using a number of days to be predicted that is smaller than or equal to the number of days that were used.

* **date_begin** : This defines the date from which the model should start training. Usually this is set to 2020-02-01 because that is when the coronavirus started to arrive in Belgium.

* **epochs** :  This is the number of epochs that the model should do during the training process.

* **topics** : This represents the list of topics that should be considered by the model during the training. These topics must have been collected previously before being able to use it in the prediction.

* **scaler** : This parameters is used to determine the scaling that should be done on the data during the training. This can be a _MinMax_ or a _Standard_ scaler.

* **predict_one** : This parameter can be set to _true_ in order to generate a unique prediction at days_to_predict days ahead. If it is set to _false_, the prediction will be done for each day between today and today +  days_to_predict days.

* **cumsum** : This parameter can be set to _true_ if we need to consider the cumulative sum of the number of new hospitalizations. It is set to _false_ if the goal is to consider the absolute number of hospitalizations.

* **verbose** :  This parameter can be set to _true_ if the user wants to print the evolution of the training process and is set to _false_ otherwise.


### Prediction of the number of hospitalizations with an untrainable model : 

* **name** : This represents the name of the untrainable model that should be used for the prediction task. It can be _baseline_ or _linear-regression_

* **days_to_use** : This represents the number of days that should be used by the model to make the prediction. 

* **days_to_predict** : This parameter is used to determine the number of days that should be predicted by the model.

If you want to make the prediction with a trainable model, you do not need to specify the parameters in the _untrainable_ section in the config file and vice-versa. 