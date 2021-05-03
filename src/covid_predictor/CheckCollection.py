from pydantic import BaseModel
from enum import Enum
from datetime import date
from typing import List


class TargetEnum(Enum):
    tot_hosp = "total hospi"
    new_hosp = "new hospi"
    perc_change = "perc change"
    log_hosp = "log hosp"
    

class NameEnum(Enum):
    name1 = "collection"
    name2 = "collect"  


class TopicsEnum(Enum):
    topic1 = "Fièvre"
    topic2 = "Mal de gorge"
    topic3 = "Toux"
    topic4 = "Virus"
    topic5 = "Symptôme"
    topic6 = "Respiration"
    topic7 = "Dyspnée"
    topic8 = "Agueusie"
    topic9 = "Anosmie"
    topic10 = "Épidémie"
    

class CollectionEnum(Enum):
    daily = "daily" 
    hourly = "hourly"


class CheckCollection(BaseModel):
    name: NameEnum
    collection: CollectionEnum
    topics: List[TopicsEnum]
    european_geo: bool
