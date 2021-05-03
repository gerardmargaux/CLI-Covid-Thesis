from pydantic import BaseModel, Field, conint, root_validator
from enum import Enum
from datetime import date
from decimal import Decimal
from typing import List, Dict, Literal


class TargetEnum(Enum):
    tot_hosp = "total hospi"
    new_hosp = "new hospi"
    perc_change = "perc change"
    log_hosp = "log hosp"
    

class ModelEnum(Enum):
    encoder_decoder = "encoder-decoder"
    assembler = "assembler"
    linear_regression = "linear-regression"
    baseline = "baseline"
    dense = "dense"
    

class NameEnum(Enum):
    name1 = "prediction"
    name2 = "predict"   
    name3 = "pred"


class CheckPrediction(BaseModel):
    name: NameEnum
    days_to_use: conint(gt=0, lt=30)
    days_to_predict: conint(gt=0, lt=20)
    predict_one: bool
    target: TargetEnum
    model: ModelEnum
    date_begin: date
    
    @root_validator()
    def check_days(cls, values):
        if values.get("days_to_use") < values.get("days_to_predict"):
            raise ValueError('days_to_use should be greater or equal to days_to_predict')
        return values


    

