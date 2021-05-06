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

* **collection** : Refers to the type of collection that should be done. Can be _hourly_ to collect Trends data hour-by-hour or _daily_ to collect data day-by-day.

* **topics** : List of topics that will be used to collect data. 

* **european_geo** : is _true_ if the european geocodes should be considered. Is _false_ if the localisation that should be considered is Belgium + some French regions.

### Training of the model :

### Prediction of the number of hospitalizations :