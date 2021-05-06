from pydantic import BaseModel, Field, conint, root_validator
from enum import Enum
from datetime import date
from decimal import Decimal
from typing import List, Dict, Literal


class TargetEnum(Enum):
    tot_hosp = 'TOT_HOSP'
    new_hosp = 'NEW_HOSP'
    perc_change = 'TOT_HOSP_pct'
    log_hosp = 'TOT_HOSP_log'
    

class CollectionEnum(Enum):
    daily = "daily" 
    hourly = "hourly"
    

class NameTrainEnum(Enum):
    name1 = "training"
    name2 = "train"   
    

class ModelTrainableEnum(Enum):
    encoder_decoder = "encoder-decoder"
    simple_autoencoder = "simple-auto-encoder"
    assembler = "assembler"
    dense = "dense"
    

class ModelUntrainableEnum(Enum):
    baseline = "baseline"
    linear_regression = "linear-regression"
    

class NamePredEnum(Enum):
    name1 = "prediction"
    name2 = "predict"   
    name3 = "pred"
    

class NameCollectEnum(Enum):
    name1 = "collection"
    name2 = "collect"  
    
    
class TypeModelEnum(Enum):
    trainable = "trainable"
    untrainable = "untrainable"
    

class ScalerEnum(Enum):
    minmax = "MinMax"
    stand = "Standard"


class CheckTrainable(BaseModel):
    type_name: TypeModelEnum
    name: ModelTrainableEnum
    date_begin: date
    days_to_use: conint(gt=0, lt=40)
    days_to_predict: conint(gt=0, lt=25)
    epochs: conint(gt=200, lt=10000)
    topics: List[str]
    scaler: ScalerEnum
    predict_one: bool
    cumsum: bool
    verbose: bool 
       
    """@root_validator()
    def check_days(cls, values):
        if values.get("date_begin") >= values.get("date_end"):
            raise ValueError('date_begin should be before date_end')
        return values"""
    

class CheckUntrainable(BaseModel):
    type_name: TypeModelEnum
    name: ModelUntrainableEnum
    days_to_use: conint(gt=0, lt=20)
    days_to_predict: conint(gt=0, lt=25)
    

class CheckGlobalTraining(BaseModel):
    name: str
    model_type: TypeModelEnum
    target: TargetEnum
    localisation: str
    european_geo: bool
    print_results_on_terminal: bool
    plot: bool
    
    
class CheckCollection(BaseModel):
    name: NameCollectEnum
    collection: CollectionEnum
    topics: List[str]
    european_geo: bool
