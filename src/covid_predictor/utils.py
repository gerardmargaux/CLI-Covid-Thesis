# hold constant and utility functions
from typing import List, Tuple, Dict, Iterable, Union, Iterator
import pycountry
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
from copy import deepcopy
import json
from time import sleep
from calendar import monthrange
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools
from functools import reduce, partial
from pathlib import Path
import requests
from requests.exceptions import ReadTimeout
import pytrends
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq
from pytrends.dailydata import get_daily_data
import io
import operator
import random
import os.path
from os import listdir

european_geocodes = {
    'AT': 'Austria',
    'BE': 'Belgium',
    'BG': 'Bulgaria',
    'CY': 'Cyprus',
    'CZ': 'Czechia',
    'DE': 'Germany',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'ES': 'Spain',
    'FI': 'Finland',
    'FR': 'France',
    'GB': 'Great Britain',
    'GR': 'Greece',
    'HR': 'Croatia',
    'HU': 'Hungary',
    'IS': 'Iceland',
    'IE': 'Ireland',
    'IT': 'Italy',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'LV': 'Latvia',
    'MT': 'Malta',
    'NL': 'Netherlands',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'SE': 'Sweden',
    'SI': 'Slovenia',
    'SK': 'Slovakia',
}

french_region_and_be = {
    'FR-A': "Alsace-Champagne-Ardenne-Lorraine",
    'FR-B': "Aquitaine-Limousin-Poitou-Charentes",
    'FR-C': "Auvergne-Rhône-Alpes",
    'FR-P': "Normandie",
    'FR-D': "Bourgogne-Franche-Comté",
    'FR-E': 'Bretagne',
    'FR-F': 'Centre-Val de Loire',
    'FR-G': "Alsace-Champagne-Ardenne-Lorraine",
    'FR-H': 'Corse',
    'FR-I': "Bourgogne-Franche-Comté",
    'FR-Q': "Normandie",
    'FR-J': 'Ile-de-France',
    'FR-K': 'Languedoc-Roussillon-Midi-Pyrénées',
    'FR-L': "Aquitaine-Limousin-Poitou-Charentes",
    'FR-M': "Alsace-Champagne-Ardenne-Lorraine",
    'FR-N': 'Languedoc-Roussillon-Midi-Pyrénées',
    'FR-O': 'Nord-Pas-de-Calais-Picardie',
    'FR-R': 'Pays de la Loire',
    'FR-S': 'Nord-Pas-de-Calais-Picardie',
    'FR-T': "Aquitaine-Limousin-Poitou-Charentes",
    'FR-U': "Provence-Alpes-Côte d'Azur",
    'FR-V': "Auvergne-Rhône-Alpes",
    'BE': "Belgique"
}

european_adjacency = [
    ('AT', 'CZ'),
    ('AT', 'DE'),
    ('AT', 'HU'),
    ('AT', 'IT'),
    ('AT', 'SI'),
    ('AT', 'SK'),
    ('BE', 'DE'),
    ('BE', 'FR'),
    ('BE', 'LU'),
    ('BE', 'NL'),
    ('BG', 'GR'),
    ('BG', 'RO'),
    # CY is an island -> no neighbor
    ('CZ', 'DE'),
    ('CZ', 'PL'),
    ('CZ', 'SK'),
    ('DE', 'DK'),
    ('DE', 'FR'),
    ('DE', 'LU'),
    ('DE', 'NL'),
    ('DE', 'PL'),
    ('DK', 'SE'),  # Denmark is considered to be adjacent to Sweden
    ('EE', 'LV'),
    ('ES', 'FR'),
    ('ES', 'PT'),
    ('FI', 'NO'),
    ('FI', 'SE'),
    ('FR', 'IT'),
    ('FR', 'LU'),
    ('HR', 'HU'),
    ('HR', 'SI'),
    ('HU', 'RO'),
    ('HU', 'SI'),
    ('HU', 'SK'),
    # no neighbor for Iceland and for Ireland
    ('IT', 'SI'),
    ('LT', 'LV'),
    ('LT', 'PL'),
    # malta is an island -> no neighbor
    ('NO', 'SE'),
    ('PL', 'SK'),
]

france_region_adjacency = [
    ('FR-A', 'FR-M'),
    ('FR-A', 'FR-I'),
    ('FR-B', 'FR-T'),
    ('FR-B', 'FR-L'),
    ('FR-B', 'FR-N'),
    ('FR-C', 'FR-F'),
    ('FR-C', 'FR-D'),
    ('FR-C', 'FR-V'),
    ('FR-C', 'FR-K'),
    ('FR-C', 'FR-N'),
    ('FR-C', 'FR-L'),
    ('FR-D', 'FR-J'),
    ('FR-D', 'FR-G'),
    ('FR-D', 'FR-I'),
    ('FR-D', 'FR-V'),
    ('FR-D', 'FR-F'),
    ('FR-E', 'FR-P'),
    ('FR-E', 'FR-R'),
    ('FR-F', 'FR-Q'),
    ('FR-F', 'FR-J'),
    ('FR-F', 'FR-L'),
    ('FR-F', 'FR-T'),
    ('FR-F', 'FR-R'),
    ('FR-F', 'FR-P'),
    ('FR-G', 'FR-M'),
    ('FR-G', 'FR-I'),
    ('FR-G', 'FR-J'),
    ('FR-G', 'FR-S'),
    # no adjacent region for FR-H
    ('FR-I', 'FR-M'),
    ('FR-I', 'FR-V'),
    ('FR-J', 'FR-S'),
    ('FR-J', 'FR-Q'),
    ('FR-K', 'FR-V'),
    ('FR-K', 'FR-U'),
    ('FR-K', 'FR-N'),
    ('FR-L', 'FR-N'),
    ('FR-L', 'FR-T'),
    ('FR-O', 'FR-S'),
    ('FR-P', 'FR-Q'),
    ('FR-P', 'FR-R'),
    ('FR-Q', 'FR-S'),
    ('FR-R', 'FR-T'),
    ('FR-U', 'FR-V'),
]

# source: https://en.wikipedia.org/wiki/List_of_European_countries_by_population (UN estimate)
european_population = {
    'AT':  9_006_398,
    'BE': 11_589_623,
    'BG':  6_948_445,
    'CY':  1_195_750,
    'CZ': 10_729_333,
    'DE': 83_783_942,
    'DK':  5_805_607,
    'EE':  1_330_299,
    'ES': 46_811_531,
    'FI':  5_548_480,
    'FR': 65_273_511,
    'GR': 10_391_029,
    'HR':  4_086_308,
    'HU':  9_646_008,
    'IS':    343_008,
    'IE':  4_992_908,
    'IT': 60_461_826,
    'LT':  2_690_259,
    'LU':    635_755,
    'LV':  1_870_386,
    'MT':    514_564,
    'NL': 17_161_189,
    'NO':  5_449_099,
    'PL': 37_830_336,
    'PT': 10_175_378,
    'RO': 19_126_264,
    'SE': 10_147_405,
    'SI':  2_080_044,
    'SK':  5_463_818,
}


def log_values(df: pd.DataFrame, columns: list = None, base: int = 10, inf_value='drop') -> pd.DataFrame:
    """
    add log values to the dataframe
    :param df: dataframe to change
    :param columns: list of name of the columns that needs to be modified. None= all columns
    :param base: base for the logarithm. Supported: [10]. If not in the list, use logarithm in base e
    :param inf_value: value to give for the inf created by the log. Can be integer or 'drop' (dropping the values)
    :return dataframe with log values for the corresponding columns
    """
    if columns == None:
        columns = df.columns
    new_columns = [f"{name}_log" for name in columns]

    if base == 10:
        df[new_columns] = np.log10(df[columns])
    else:
        df[new_columns] = np.log(df[columns]) / np.log(base)

    if inf_value == 'drop':
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    else:  # inf_value should be an integer
        df = df.replace([np.inf, -np.inf], inf_value)
    return df


def pct_values(df: pd.DataFrame, columns: list = None, add_one: bool = False) -> pd.DataFrame:
    """
    add percentage values to the dataframe
    :param df: dataframe to change
    :param columns: list of name of the columns that needs to be modified. None= all columns
    :param add_one: if True, the percentage of difference add a value of 100% to each percentage
    :return dataframe with pct change values for the corresponding columns
    """
    if columns == None:
        columns = df.columns
    new_columns = [f"{name}_pct" for name in columns]
    df[new_columns] = df[columns].pct_change().replace([np.nan, np.inf], 0)
    if add_one:
        df[new_columns] = df[new_columns] + 1
    return df


def get_world_population(pop_file: str, alpha2: bool = True) -> Dict[str, float]:
    """
    :param pop_file: path to the population file, registered as a dict
    :param alpha2: whether to return the dict with the keys being alpha 2 coded or not
    :return dict of population
    """
    pop = json.load(open(pop_file))
    if alpha2:
        return {alpha3_to_alpha2(k): v for k, v in pop.items() if len(k) == 3}
    else:
        return pop


def hospi_french_region_and_be(hospi_france_tot, hospi_france_new, hospi_belgium, department_france, geo,
                               new_hosp=True, tot_hosp=True, new_hosp_in=False, date_begin: str = None):
    """
    Creates the dataframe containing the number of daily hospitalizations in Belgium and in the french regions
    with respect to the date and the localisation (FR and BE)
    :param hospi_france_tot: url/path for the total french hospitalisations csv
    :param hospi_france_new: url/path for the new french hospitalisations csv
    :param hospi_belgium: url/path for the belgian hospitalisations csv
    :param department_france: url/path for the mapping of french department to regions
    :param geo: geocode of the region that should be incuded in the final dict
    :param new_hosp_in: if True, includes the new daily hospitalisations (inwards)
    :param tot_hosp: if True, includes the total hospitalisations
    :return dict of {geocode: hosp_df} where hosp is the hospitalisation dataframe of each geocode
    """
    columns_be = {}  # only for belgium, not for france (the files are handled differently)
    data_columns = []  # final data columns that will be present in the df
    if new_hosp_in:
        columns_be['NEW_IN'] = 'sum'
        data_columns.append("NEW_HOSP_IN")
    if tot_hosp:
        columns_be['TOTAL_IN'] = 'sum'
        data_columns.append("TOT_HOSP")
    if len(columns_be) == 0:
        raise Exception("no hospitalisation column specified")
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    departements = pd.read_csv(department_france)

    # French data: total hospitalisation
    if tot_hosp or new_hosp:
        hospitalisations = pd.read_csv(hospi_france_tot, sep=";", parse_dates=['jour'], date_parser=date_parser)
        hospitalisations = hospitalisations[hospitalisations['sexe'] == 0]  # sex=0: men and women
        data_fr_tot = hospitalisations.join(departements.set_index('departmentCode'), on="dep").groupby(
            ["regionTrends", "jour"], as_index=False).agg({"hosp": "sum"})

    # French data: new hospitalisation
    if new_hosp_in:
        hospitalisations = pd.read_csv(hospi_france_new, sep=";", parse_dates=['jour'], date_parser=date_parser)
        data_fr_new = hospitalisations.join(departements.set_index('departmentCode'), on="dep").groupby(
            ["regionTrends", "jour"], as_index=False).agg({"incid_hosp": "sum"})

    # merge the french data
    common_columns = ["regionTrends", "jour"]
    if (tot_hosp or new_hosp) and new_hosp_in:
        data_fr = data_fr_tot.merge(data_fr_new, how='outer', left_on=common_columns, right_on=common_columns).fillna(0)
    elif tot_hosp or new_hosp:
        data_fr = data_fr_tot
    elif new_hosp_in:
        data_fr = data_fr_new
    data_fr = data_fr.rename(
        columns={"jour": "DATE", "regionTrends": "LOC", "hosp": "TOT_HOSP", "incid_hosp": "NEW_HOSP_IN"})

    # Belgian data
    data_be = pd.read_csv(hospi_belgium, parse_dates=['DATE'], date_parser=date_parser).groupby(
        ["DATE"], as_index=False).agg(columns_be).rename(
        columns={"TOTAL_IN": "TOT_HOSP", "NEW_IN": "NEW_HOSP_IN"})
    data_be["LOC"] = "BE"

    # Full data
    full_data = data_fr.append(data_be).set_index(["LOC", "DATE"])

    # find smallest date for each loc and highest common date
    smallest = {}
    highest = {}
    for loc, date_current in full_data.index:
        if loc not in smallest or smallest[loc] > date_current:
            smallest[loc] = date_current
        if loc not in highest or highest[loc] < date_current:
            highest[loc] = date_current

    highest_date = min(highest.values())
    base_date = datetime.strptime(date_begin, "%Y-%m-%d").date()

    # Add "fake" data (zeroes before the beginning of the crisis) for each loc
    toadd = []
    add_entry = [0 for i in range(len(data_columns))]  # each missing entry consist of zero for each data col
    for loc, sm in smallest.items():
        end = sm.date()
        cur = base_date
        while cur != end:
            toadd.append([cur, loc, *add_entry])
            cur += timedelta(days=1)

    full_data = pd.DataFrame(toadd, columns=["DATE", "LOC", *data_columns]).append(full_data.reset_index()).set_index(
        ["LOC", "DATE"])
    data_dic = {}

    for k, v in geo.items():
        data_dic[k] = full_data.iloc[(full_data.index.get_level_values('LOC') == k) &
                                     (full_data.index.get_level_values('DATE') <= highest_date)]
        if new_hosp:
            data_dic[k]['NEW_HOSP'] = data_dic[k]['TOT_HOSP'].diff()
            data_dic[k].at[data_dic[k].index.min(), 'NEW_HOSP'] = 0
    return data_dic


def hospi_world(hospi_file: str, geo: Dict[str, str], renaming: Dict[str, str], date_begin: str,
                tot_hosp=True, new_hosp=False, tot_icu=False, new_icu=False) -> Dict[str, pd.DataFrame]:
    """
    Creates the dataframe containing the number of daily hospitalisations in Europe and
    update the geocodes given in order to remove regions without data
    :param hospi_file: url/path for the hospitalisations csv
    :param geo: geocode of the countries that should be incuded in the final dict. The dict is updated if a
        region does not have data
    :param renaming: renaming to use for the countries
    :param date_begin: date of beginning (format YYYY-MM-DD), 0 will be added from this date until the first date where
        data can be found
    :param tot_hosp: whether or not to give the total hosp in the final df
    :param new_hosp: whether or not to give the new hosp in the final df
    :param tot_icu: whether or not to give the total icu in the final df
    :param new_icu: whether or not to give the new icu in the final df
    :return dict of {geocode: hosp_df} where hosp is the hospitalisation dataframe of each geocode
    """
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    full_data = pd.read_csv(hospi_file, parse_dates=['date'], date_parser=date_parser).rename(
        columns={'iso_code': 'LOC', 'date': 'DATE', 'icu_patients': 'TOT_ICU', 'hosp_patients': 'TOT_HOSP'})
    # transform iso code from alpha 3 to alpha 2
    code_mapping = {}
    for code in full_data['LOC'].unique():
        if len(code) == 3:
            code_mapping[code] = alpha3_to_alpha2(code)

    full_data = full_data.replace({**renaming, **code_mapping})
    full_data = full_data.set_index(["LOC", "DATE"])

    data_columns = []
    if tot_icu:
        data_columns.append('TOT_ICU')
    if tot_hosp:
        data_columns.append('TOT_HOSP')
    full_data = full_data[data_columns]

    if new_hosp:
        data_columns.append('NEW_HOSP')
    if new_icu:
        data_columns.append('NEW_ICU')
    add_entry = [0 for _ in range(len(data_columns))]
    data_dic = {}
    base_date = datetime.strptime(date_begin, "%Y-%m-%d").date()
    country_to_remove = []

    for loc in geo:
        df = full_data.iloc[full_data.index.get_level_values('LOC') == loc]
        # reindex in case missing values without NaN appear
        min_date = df.index.get_level_values('DATE').min()
        max_date = df.index.get_level_values('DATE').max()
        reindexing = pd.MultiIndex.from_product([[loc], pd.date_range(min_date, max_date)], names=['LOC', 'DATE'])
        df = df.reindex(reindexing, fill_value=np.nan)
        df = df.interpolate(limit_area='inside').dropna()
        # remove NaN entries
        df = df.dropna()
        # the dataframe might not have any entry at this point -> remove it if it the case
        if df.empty:
            print(f"region {loc} does not have any entry, removing it")
            country_to_remove.append(loc)
            continue
        # add zeros at the beginning if no data is found
        smallest_date = df.index.get_level_values('DATE').min()

        to_add = []
        end = smallest_date
        cur = base_date
        if cur <= end:
            while cur != end:
                to_add.append([cur, loc, *add_entry])
                cur += timedelta(days=1)
            df = pd.DataFrame(to_add, columns=["DATE", "LOC", *data_columns]).append(df.reset_index()).set_index(
                ["LOC", "DATE"])
        else:  # drop data if it is too early
            begin = datetime.fromordinal(base_date.toordinal())
            df = df.iloc[df.index.get_level_values('DATE') >= begin]
        # add the relevant new columns
        if new_icu:
            df['NEW_ICU'] = df['TOT_ICU'].diff()
            df.at[df.index.min(), 'NEW_ICU'] = 0
        if new_hosp:
            df['NEW_HOSP'] = df['TOT_HOSP'].diff()
            df.at[df.index.min(), 'NEW_HOSP'] = 0
        data_dic[loc] = df
    for loc in country_to_remove:
        del geo[loc]
    return data_dic


def create_df_trends(url_trends: str, list_topics: Dict[str, str], geo: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    return dic of {geo: df} for the trends
    :param url_trends: path to the trends data folder
    :param list_topics: dict of topic title: topic code for each google trends
    :param geo: dict of geo localisations to use
    """
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    renaming = {v: k for k, v in list_topics.items()}  # change topic_mid to topic_title in the dataframe
    if len(renaming) == 0:
        return {k: pd.DataFrame() for k in geo}
    result = {}
    for k, v in geo.items():
        all_trends = []
        for term in list_topics.keys():
            path = f"{url_trends}{k}-{term}.csv"
            if url_trends[:4] == "http":
                encoded_path = requests.get(path).content
                df_trends = pd.read_csv(io.StringIO(encoded_path.decode("utf-8")), parse_dates=['date'],
                                        date_parser=date_parser).rename(columns={"date": "DATE"})
            else:
                df_trends = pd.read_csv(path, parse_dates=['date'], date_parser=date_parser).rename(columns={"date": "DATE"})
            df_trends['LOC'] = k
            df_trends.rename(columns=renaming, inplace=True)
            df_trends.set_index(['LOC', 'DATE'], inplace=True)
            all_trends.append(df_trends)
        result[k] = pd.concat(all_trends, axis=1)
    return result


def world_adjacency_list(alpha: int = 3) -> List[Tuple[str, str]]:
    """
    :param alpha: iso code to use. Must be in [2, 3]
    :return list of adjacent countries. Neighboring is listed only once, not twice.
    """
    #df = pd.read_csv("https://raw.githubusercontent.com/geodatasource/country-borders/master/GEODATASOURCE-COUNTRY-BORDERS.CSV").dropna()
    df = pd.read_csv("../data/country_borders.csv").dropna()
    adj_list = []
    for idx, row in df.iterrows():
        if row['country_code'] < row['country_border_code']:  # list neighboring once only
            adj_list.append((row['country_code'], row['country_border_code']))
    if alpha == 2:
        return adj_list
    elif alpha == 3:
        return [(alpha2_to_alpha3(a), alpha2_to_alpha3(b)) for a, b in adj_list]
    else:
        raise ValueError("alpha must be 2 or 3")


def alpha2_to_alpha3(code):  # transform a country code from alpha 2 to alpha 3
    return pycountry.countries.get(alpha_2=code).alpha_3


def alpha3_to_alpha2(code):  # transform a country code from alpha 3 to alpha 2
    return pycountry.countries.get(alpha_3=code).alpha_2


def add_lag(df, lag, dropna=True):
    """
    add lagged columns to a dataframe
    :param df: dataframe to modify
    :param lag: lag to add. Values can be negative (old values) or positive (forecast values)
        if positive, lag values from the future are added, excluding the ones from today
        otherwise, lag-1 values from the past are added, including the ones from today
    :param dropna: if True, drop the NaN columns created
    """
    if lag < 0:  # add values from the past
        lag_range = range(lag + 1, 0, 1)
    else:  # add values from the future
        lag_range = range(1, lag + 1, 1)
    columns = []

    init_names = df.columns
    for i in lag_range:
        renaming = {col: f"{col}(t{i:+d})" for col in init_names}  # name of the lagged columns
        columns.append(df.shift(-i).rename(columns=renaming))

    if lag < 0:
        columns.append(df)  # include the original data if lag < 0
    df = pd.concat(columns, axis=1)
    return df.dropna() if dropna else df


def region_merge_iterator(init_regions: List[str], nb_merge: int, adj: List[Tuple[str, str]] = None):
    """
    yield list of regions, supposed to be used to form augmented regions
    :param init_regions: list of regions to merge
    :param nb_merge: number of regions that can be used (at most) to create an augmented region. Must be >=2
    :param adj: list of adjacency to use. If None, all regions will be mixed, even unadjacent. Otherwhise
        use the list of adjacent region to augment the data
    :return: yield list of regions to merge
    """
    if adj:
        G = nx.Graph(adj)
    for merge in range(2, nb_merge+1):
        for elem in itertools.combinations(init_regions, merge):
            if adj:  # check if the region candidate can be formed
                connected = [False for _ in range(len(elem))]
                for i, node_a in enumerate(elem):
                    for j, node_b in enumerate(elem[i+1:]):
                        if G.has_edge(node_a, node_b):
                            connected[i] = True
                            connected[i + j + 1] = True
                if np.all(connected):
                    yield elem
            else:  # unadjacent regions can be given
                yield elem
                  

def dates_iterator(begin: datetime, end: datetime, number: int) -> Iterator[Tuple[datetime, datetime]]:
    """
    return the largest interval of dates that must be available to provide number queries on the interval
    :param begin: date of beginning for the query
    :param end: date of end for the query
    :param number: amount of queries that must cover this interval
    :return yield number tuples of dates (a, b), that cover [begin, end]
    """
    # maximum date allowed for google trends, included
    max_end = latest_trends_daily_date()
    number_given = 0
    lag_left = 0
    lag_right = 0
    if end > max_end:  # impossible to provide the queries
        return []
    while number_given < number:
        i = 0
        while number_given < number and i <= lag_right:
            wanted_end = end + timedelta(days=i)
            if wanted_end <= max_end:
                number_given += 1
                yield begin - timedelta(days=lag_left), wanted_end
            i += 1

        wanted_end = end + timedelta(days=lag_right)
        if wanted_end <= max_end:
            j = lag_left - 1
            while number_given < number and j >= 0:
                number_given += 1
                yield begin - timedelta(days=j), wanted_end
                j -= 1

        lag_left += 1
        lag_right += 1


def mean_query(number: int, begin: datetime, end: datetime, topic: str, geo: str, cat=0, verbose=True):
    """
    provide multiple queries on the period begin->end. the column topic contains the mean of the queries
    the queries use different interval in order to provide different results
    """
    df_tot = pd.DataFrame()
    cnt = 0
    pytrends = TrendReq(retries=2, backoff_factor=0.1)
    print(begin, end)
    for k, (begin_tmp, end_tmp) in enumerate(dates_iterator(begin, end, number)):
        timeframe = dates_to_timeframe(begin_tmp, end_tmp)
        # Initialize build_payload with the word we need data for
        try:
            current_ip = tor_ip_changer.get_new_ip()
        except:
            pass
        build_payload = partial(pytrends.build_payload,
                                kw_list=[topic], cat=cat, geo=geo, gprop='')
        if verbose:
            print(f"timeframe= {timeframe} ({k + 1}/{number})")
        df = _fetch_data(pytrends, build_payload, timeframe)
        df = df[begin:end]
        if 100 not in df[topic]:
            df[topic] = df[topic] * 100 / df[topic].max()
        df_tot[f"{topic}_{cnt}"] = df[topic]
        cnt += 1
        if cnt >= number:
            df_tot[topic] = df_tot.mean(axis=1)
            df_tot[topic] = 100 * df_tot[topic] / df_tot[topic].max()
            return df_tot


def timeframe_normalize_clusters(list_df, overlap=30):
    """
    take as input the list of df given by sanitize_hourly_data and return the list of dates needed to normalize
    the whole set of data, as tuple of dates
    """
    list_tf = []
    delta = timedelta(days=overlap)
    max_end = latest_trends_daily_date()
    for df_i, df_j in zip(list_df, list_df[1:]):
        begin = (df_i.index.max() - delta).to_pydatetime()
        end = (df_j.index.min() + delta).to_pydatetime()
        if end > max_end:
            end = max_end
        list_tf.append((begin, end))
    return list_tf


def drop_incomplete_days(list_df: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    filter a list of hourly dataframes, to return a list where each dataframe:
    - begins at MM-DD-0h
    - ends at MM-DD-23h
    - contains at least 3 days of data
    - has at least 3 points of value > 10
    - has less than 10 consecutive values of 0 at the end / beginning
    :param list_df: list of dataframes to filter
    """
    result = []
    for i in range(len(list_df)):
        df = list_df[i]
        old_begin, old_end = df.index.min(), df.index.max()
        new_begin = old_begin + timedelta(hours=((24 - old_begin.hour) % 24))  # MM-DD-0h
        new_end = old_end - timedelta(hours=((old_end.hour + 1) % 24))  # MM-DD-23h
        cur_df = df[new_begin:new_end]
        # check for chain of zeros at the beginning and the end
        has_zero = True
        hours_drop = 10  # consecutive values to check
        delta = timedelta(hours=hours_drop)
        while has_zero and new_begin < new_end:  # zeros at the beginning
            if cur_df[new_begin:new_begin + delta].sum()[0] == 0:
                new_begin += timedelta(days=1)
            else:
                has_zero = False

        has_zero = True
        while has_zero and new_begin < new_end:  # zeros at the end
            if cur_df[new_end - delta:new_end].sum()[0] == 0:
                new_end -= timedelta(days=1)
            else:
                has_zero = False
        # new dates for the dataframe
        cur_df = cur_df[new_begin:new_end]
        # check if the resulting dataframe can be added
        if not cur_df.empty and (new_end - new_begin).days >= 2 and len(np.where(cur_df > 10)[0]) > 3:
            result.append(cur_df)
    return result


def sanitize_hourly_data(df: pd.DataFrame, topic_code: str) -> List[pd.DataFrame]:
    """
    sanitize hourly data, transforming it to a list of daily data and removing missing values
    :param df: dataframe of hourly data on a trends topic, indexed by hourly dates
    :param topic_code: code for the topic
    :return list of data sanitized: missing values are removed, leading to dataframes with holes between one another
    """
    list_df_hourly = scale_df(df, topic_code)  # scale the dataframe
    list_df_hourly = drop_incomplete_days(list_df_hourly)  # drop the incomplete days (check doc for details)
    list_df_hourly = [df.resample('D').mean() for df in list_df_hourly]  # aggregate to daily data
    return list_df_hourly


def find_largest_intersection(df_a: pd.DataFrame, df_b: pd.DataFrame, list_df_daily: List[pd.DataFrame],
                              overlap: int = 30) \
        -> Tuple[pd.DataFrame, bool]:
    """
    find daily dataframe with the largest intersection on df_a and df_b
    :param df_a: first dataframe
    :param df_b: second dataframe
    :param list_df_daily: list of dataframe to consider in order to find the one with the largest intersection
    :param overlap: number of overlap that should be used for the intersection
    """
    if not list_df_daily:  # no list of daily dataframe given
        return pd.DataFrame(), True

    best_inter = -1
    best_df = None
    can_be_actualized = True  # true if the largest date must be actualized
    max_date = latest_trends_daily_date()
    max_overlap_left = min(len(df_a), overlap)  # upper bound considered for the overlap
    max_overlap_right = min(len(df_b), overlap)
    for df_candidate in list_df_daily:
        intersection_left = len(df_a.index.intersection(df_candidate.index))
        intersection_right = len(df_b.index.intersection(df_candidate.index))
        inter = min(intersection_left, intersection_right)
        if inter >= best_inter:
            best_df = df_candidate
            best_inter = inter
            if intersection_right < max_overlap_right and df_candidate.index.max() < max_date:  # new data is available
                can_be_actualized = True
            elif intersection_left < max_overlap_left:  # better data can be found
                can_be_actualized = True
            else:
                can_be_actualized = False
    return best_df, can_be_actualized


def daily_gap_and_model_data(geo: str, topic_title: str, topic_mid: str, number: int = 20, overlap: int = 30,
                             verbose: bool = True, refresh: bool = True) -> pd.DataFrame:
    """
    collect the daily data on the gap where the hourly data could not be retrieved
    if the daily data is outdated or absent, it is collected and saved to a csv file
    :param geo: geocode that must be queried
    :param topic_title: title of the topic to query
    :param topic_mid: code of the topic to query
    :param number: number of daily queries to do on the gaps
    :param overlap: number of overlapping points between a daily request and the 2 hourly requests around it
    :param verbose: whether to display information while running or not
    :param refresh: whether to find new daily gap or use only the existing ones
    :return model dataframe
    """
    data_daily_dir = "../data/trends/collect_gap"
    data_hourly_dir = "../data/trends/collect"
    data_model_dir = "../data/trends/model"
    # retrieve the hourly data
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df_hourly = pd.read_csv(f"{data_hourly_dir}/{geo}-{topic_title}.csv", parse_dates=['date'],
                            date_parser=date_parser).set_index('date')
    # transform the hourly data to a list of daily data
    list_df_hourly = sanitize_hourly_data(df_hourly, topic_mid)
    # collect the existing daily data
    # pattern = "([A-Z]{2}(?:-[A-Z])?)-(\D*)-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})\.csv"
    starting_pattern = f"{geo}-{topic_title}-"
    existing_files = [filename for filename in os.listdir(data_daily_dir) if filename.startswith(starting_pattern)]
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    list_daily_df = [pd.read_csv(f"{data_daily_dir}/{file}", parse_dates=['date'],
                                 date_parser=date_parser).set_index('date')[[topic_mid]] for file in existing_files]
    if verbose:
        for i, df in enumerate(list_df_hourly):
            print(
                f"hourly agg data: cluster {i}: {df.index.min().to_pydatetime().date()} -> {df.index.max().to_pydatetime().date()}")
    # check if the gaps can be covered with the existing daily requests saved
    list_dates_actualize = []
    list_daily_intersection = []
    for df_a, df_b in zip(list_df_hourly, list_df_hourly[1:]):
        df_intersection, can_be_actualized = find_largest_intersection(df_a, df_b, list_daily_df, overlap=overlap)
        if can_be_actualized and refresh:  # new data can be queried and must be refreshed
            # overlap-1 because the date is included
            dates_query = (df_a.index.max() - timedelta(days=overlap - 1)).to_pydatetime(), \
                          min((df_b.index.min() + timedelta(days=overlap - 1)).to_pydatetime(),
                              latest_trends_daily_date())
            list_dates_actualize.append(dates_query)
        else:  # the query respect the expected overlap
            list_daily_intersection.append(df_intersection)
    files_to_remove = []
    for i, df in enumerate(list_daily_df):
        to_remove = True
        for df_chosen in list_daily_intersection:
            if df is df_chosen:
                to_remove = False
                break
        if to_remove:
            files_to_remove.append(existing_files[i])

    # do the request for the dataframes to actualize
    min_dates = find_min_dates_queries(list_dates_actualize, number)
    if min_dates and verbose:
        print("querying new interval")
    for begin, end in min_dates:
        df = collect_holes_data(topic_mid, topic_title, number, begin, end, geo, cat=0, verbose=verbose)
        list_daily_intersection.append(df[[topic_mid]])

    # remove the old daily dataframes that are not used anymore
    for file in files_to_remove:
        if verbose:
            print(f"removing {data_daily_dir}/{file}")
        os.remove(f"{data_daily_dir}/{file}")

    # form the model data and save it to csv
    complete_df = merge_hourly_daily(list_df_hourly, list_daily_intersection, topic_mid, drop=True)
    filename = f"{data_model_dir}/{geo}-{topic_title}.csv"
    complete_df.to_csv(filename)
    return complete_df


def rescale_batch(df_left, df_right, df_daily, topic_code, overlap_max=30, rolling=7, overlap_min=2, drop=True):
    """
    rescale a left and a right batch with a hole in between, covered by df_daily
    :param df_left: DataFrame on the left interval
    :param df_right: DataFrame on the right interval
    :param df_daily: DataFrame with data between df_left and df_right, used for the merge
    :param overlap_max: maximum number of datapoints used for the overlap
    :param rolling: rolling average on the df_daily data
    :param overlap_min: minimum number of points on the intersection allowed to accept a rolling average.
        If the rolling average provide less points, df_daily is used instead of the rolling data.
    :param drop: whether to drop preferabily the df_daily data on the interval or not
    :return batch_rescaled: DataFrame of data between df_left.min and df_right.max
    """
    if drop:
        drop = ['right', 'left']
    else:
        drop = [None, None]

    daily_rolling = df_daily.rolling(rolling, center=True).mean().dropna()
    daily_rolling = 100 * daily_rolling / daily_rolling.max()
    overlap = df_left.index.intersection(daily_rolling.index)
    overlap_right = daily_rolling.index.intersection(df_right.index)
    # print(len(overlap))
    # print(len(overlap_right))
    if len(overlap) < overlap_min or len(overlap_right) < overlap_min:
        overlap = df_left.index.intersection(df_daily.index)
        daily_used = df_daily
    else:
        daily_used = daily_rolling

    if len(overlap) > overlap_max:
        overlap = overlap[-overlap_max:]
    overlap_len = len(overlap)
    title = daily_used.columns[0]
    daily_used.loc[daily_used[title] < 0.01, title] = 0
    batch_rescaled = merge_trends_batches(df_left, daily_used[overlap.min():], overlap_len, topic_code,
                                          is_hour=False, verbose=False, drop=drop[0])
    batch_rescaled = 100 * batch_rescaled / batch_rescaled.max()
    overlap = batch_rescaled.index.intersection(df_right.index)
    if len(overlap) > overlap_max:
        overlap = overlap[:overlap_max]
    overlap_len = len(overlap)
    batch_rescaled = merge_trends_batches(batch_rescaled[:overlap.max()], df_right, overlap_len, topic_code,
                                          is_hour=False, verbose=False, drop=drop[1])
    batch_rescaled = 100 * batch_rescaled / batch_rescaled.max()
    return batch_rescaled


def merge_hourly_daily(list_df_hourly: List[pd.DataFrame], list_df_daily: List[pd.DataFrame], topic_code: str,
                       drop: bool, add_daily_end=True):
    """
    merge the hourly (deterministic) aggregated batches, using the daily (stochastic) batches on the missing interval
    :param list_df_hourly: sorted list of deterministic DataFrame, having a daily index
    :param list_df_daily: list of stochastic DataFrame, having data on the missing interval of list_df_hourly
    :param topic_code: topic code
    :param drop: whether to drop the stochastic data preferably or not
    :param add_daily_end: if True, add daily data data at the end if the max date of daily data > max date of hourly data
    :return df: merged DataFrame
    """

    for i, df_right in enumerate(list_df_hourly):
        if i == 0:
            df = df_right
        else:
            df_daily, _ = find_largest_intersection(df, df_right, list_df_daily)
            df = rescale_batch(df, df_right, df_daily, topic_code, drop=drop)

    if add_daily_end:  # attempt to add daily data at the end
        daily_possible = [df_daily for df_daily in list_df_daily if df_daily.index.max() > df.index.max()]
        if len(daily_possible) != 0:
            column = df.columns[0]
            while len(daily_possible) > 0:
                candidate = max(daily_possible, key=lambda df_daily: df_daily.index.intersection(df.index))
                overlap_len = len(df.index.intersection(candidate.index))
                if overlap_len == 0:  # not possible to add the data since there is not overlap
                    break
                df = merge_trends_batches(df, candidate, overlap_len, column,
                                          is_hour=False, verbose=False, drop='right')
                df = df * 100 / df.max()
                daily_possible = [df_daily for df_daily in daily_possible if df_daily.index.max() > df.index.max()]
    return df


def find_min_dates_queries(list_dates: List[Tuple[datetime, datetime]], number: int) -> List[Tuple[datetime, datetime]]:
    """
    return the list of dates to query to form the minimum number of queries covering all dates provided
    uses largest_dates_iterator to determine if an interval can be queried
    :param list_dates: sorted list of tuples (begin, end) that can be queried
    :param number: number of queries that will be used to retrieve data on the interval. Used by largest_dates_iterator
    """
    if len(list_dates) == 0:
        return []
    root = (list_dates[0][0], list_dates[-1][1])
    if (root[1] - root[0]).days < max_query_days:
        return [root]
    else:
        # construct the tree
        class Node:
            def __init__(self, begin, end):
                self.begin = begin
                self.end = end
                largest_begin, largest_end = largest_dates_iterator(begin, end, number)
                self.feasible = (largest_end - largest_begin).days < max_query_days
                self.child = []
                self.parent = []
                self.covered = False

            def __str__(self):
                return str(self.begin.date()) + " " + str(self.end.date())

            def set_covered(self):
                self.covered = True
                # its 2 parents are by extension also covered
                if len(self.parent) > 0:
                    self.parent[0].set_covered()
                    self.parent[1].set_covered()

            def add_child(self, node):
                self.child.append(node)
                node.parent.append(self)

        def return_best_date(node: Node):
            queue = [node]
            dates = []
            while queue:
                node = queue.pop()
                if node.feasible and not node.covered:
                    node.set_covered()
                    dates.append((node.begin, node.end))
                elif not node.covered:
                    queue.append(node.parent[0])
                    queue.append(node.parent[1])
            return dates

        # construct the first nodes in the tree
        list_node = {0: [Node(a, b) for a, b in list_dates]}
        for depth in range(1, len(list_dates)):
            # add the child
            list_node[depth] = []
            for node_a, node_b in zip(list_node[depth - 1], list_node[depth - 1][1:]):
                node_cur = Node(node_a.begin, node_b.end)
                list_node[depth].append(node_cur)
                node_a.add_child(node_cur)
                node_b.add_child(node_cur)
        # retrieve the best interval by starting on the node at the largest depth
        best_dates = return_best_date(list_node[len(list_dates) - 1][0])
        return best_dates
    
def collect_holes_data(topic_mid: str, topic_title: str, number: int, begin: datetime, end: datetime, geo: str,
                       cat: int = 0, verbose: bool = True) -> pd.DataFrame:
    """
    collect data using number daily requests and saves it to csv
    :param topic_mid: topic code
    :param topic_title: name of the topic
    :param number: number of queries to do
    :param begin: first day that the query must cover
    :param end: last day that the query must cover
    :param geo: geocode that must be queried
    :param cat: category to filter
    :param verbose: whether to print information while collecting the data or not
    :return df_daily: data collected using n_request
    """
    dir = "../data/trends/collect_gap/"
    timeframe_included = dates_to_timeframe(begin, end).replace(" ", "-")
    filename = f"{dir}{geo}-{topic_title}-{timeframe_included}.csv"
    df_daily = mean_query(number, begin, end, topic_mid, geo=geo, cat=cat, verbose=verbose)
    if verbose:
        print(f"daily requests saved to {filename}")
    df_daily.to_csv(filename)
    return df_daily


def scale_df(df, topic):
    """
    Return a list of the scaled df. If there is always an overlap, the list contains one df.
    Otherwhise, the list contains as many df as there are clusters of periods without missing data
    Each df has its first datetime beginning at 0h and its last datetime ending at 23h
    """
    batch_id = df["batch_id"].to_list()

    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    batch_id = f7(batch_id)
    list_scaled_df = []
    scaled_df = pd.DataFrame()
    for i, j in enumerate(batch_id):
        if j < 0:  # the batch id was not valid
            if not scaled_df.empty:
                list_scaled_df.append(scaled_df)
            scaled_df = pd.DataFrame()
            continue

        batch_df = df[df["batch_id"] == j].drop(columns=["batch_id"])
        index_overlap = scaled_df.index.intersection(batch_df.index)
        overlap_hours = len(index_overlap)
        overlap_left = scaled_df.loc[index_overlap]
        overlap_right = batch_df.loc[index_overlap]
        if overlap_hours == 0 and scaled_df.empty:
            scaled_df = merge_trends_batches(scaled_df, batch_df, overlap_hours, topic)
        elif (overlap_left[topic] * overlap_right[topic]).sum() == 0:  # cannot perform the merge
            list_scaled_df.append(scaled_df)
            scaled_df = batch_df
        else:
            scaled_df = merge_trends_batches(scaled_df, batch_df, overlap_hours, topic)
    list_scaled_df.append(scaled_df)

    return list_scaled_df


def correct_batch_id(topic_title: str, topic_code: str, geo: str) -> pd.DataFrame:
    """
    correct the batch_id of the batch collected
    a batch_id must be negative if either
        - the data could not be collected on the given period
        - there is not less than 3 datapoints with value > 10
    :param topic_title: title of the topic to look at
    :param topic_code: code of the topic to look at
    :param geo: geocode of the dataframe to look at
    :return final_df: dataframe where the batch_id is set to negative if needed. This dataframe is saved in
        data/trends/collect as a csv file
    """
    data_dir = "../data/trends/collect"
    csv_file = f'{data_dir}/{geo}-{topic_title}.csv'
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(csv_file, parse_dates=['date'], date_parser=date_parser).set_index('date')
    list_batches = df.batch_id.unique()

    final_df = pd.DataFrame()
    cur_batch_id = 0  # always positive
    previous_df_batch = None
    previous_batch_id = None

    for i, batch_id in enumerate(list_batches):
        df_batch = df[df['batch_id'] == batch_id]
        # a valid batch must have at least more than 3 values above 10
        if len(np.where(df_batch[topic_code] > 10)[0]) <= 3:
            df_batch = df_batch.assign(batch_id=-cur_batch_id)
            batch_id = -cur_batch_id
        else:  # the batch is valid and must have a positive batch id
            df_batch = df_batch.assign(batch_id=cur_batch_id)
            batch_id = cur_batch_id

        # there is a gap between this batch and the previous one and the two batches don't have a negative batch_id
        if (i != 0 and len(previous_df_batch.index.intersection(df_batch.index)) == 0
                and batch_id > 0 and previous_batch_id > 0):
            min_date = previous_df_batch.index.max()
            max_date = df_batch.index.min()
            days = pd.date_range(min_date, max_date, freq='H')
            data = np.zeros(len(days))
            batch_id_vector = [-cur_batch_id for i in range(len(days))]
            df_zero = pd.DataFrame({'date': days, topic_code: data, 'batch_id': batch_id_vector})
            df_zero = df_zero.set_index('date')
            cur_batch_id += 1  # count as if 2 batches were added
            df_batch = df_zero.append(df_batch)

        previous_df_batch = df_batch
        previous_batch_id = batch_id
        final_df = final_df.append(df_batch)
        cur_batch_id += 1
    final_df.to_csv(csv_file, index=True)
    return final_df


def collect_historical_interest(topic_mid: str, topic_title: str, geo: str, begin_tot: datetime = None,
                                end_tot: datetime = None, overlap_hour: int = 15, verbose: bool = True) -> pd.DataFrame:
    """
    load collect and save hourly trends data for a given topic over a certain region
    :param topic_mid: mid code
    :param topic_title: title of the topic
    :param geo: google geocode
    :param begin_tot: beginning date. If None, default to 01/02/2020
    :param end_tot: end date. If None, default to today
    :param overlap_hour: number of overlapping point
    :param verbose: whether to print information while the code is running or not
    :return dataframe of collected data
    """
    dir = Path(__file__).parents[2]/"data/trends/collect/"
    batch_column = 'batch_id'
    hour_format = "%Y-%m-%dT%H"
    file = f"{dir}{geo}-{topic_title}.csv"
    min_delta = timedelta(days=3)
    if end_tot is None:  # if not specified, get the latest day with 24 hours for data
        end_tot = datetime.now().replace(microsecond=0, second=0, minute=0)
        if end_tot.hour != 23:
            end_tot = end_tot.replace(hour=23) - timedelta(days=1)

    if os.path.exists(file):  # load previous file
        date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        df_tot = pd.read_csv(file, parse_dates=['date'], date_parser=date_parser).set_index('date')
        idx_max = df_tot.index.max()
        if end_tot == df_tot.index.max():
            return df_tot
        max_batch = df_tot[batch_column].max()
        if len(df_tot.loc[df_tot[batch_column] == max_batch]) < 192:  # if the last batch was not done on a whole week
            df_tot = df_tot.loc[df_tot[batch_column] < max_batch]  # drop the last batch
        i = df_tot[batch_column].max() + 1  # id of the next batch
        begin_tot = df_tot.index.max() - timedelta(hours=(overlap_hour - 1))
    else:
        df_tot = pd.DataFrame()
        if begin_tot is None:
            begin_tot = datetime.strptime("2020-02-01T00", hour_format)
        i = 0

    begin_cur = begin_tot  # beginning of current batch
    end_cur = begin_tot + timedelta(days=7, hours=23)  # end of current batch
    if end_cur > end_tot:  # end of date
        end_cur = end_tot
        if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
            begin_cur = end_cur - min_delta

    delta = timedelta(days=7, hours=23) - timedelta(hours=(overlap_hour - 1))  # diff between 2 batches
    delay = 0
    trials = 0
    finished = False
    if verbose:
        print(f"topic {topic_title} geo {geo}")
    while not finished:
        timeframe = begin_cur.strftime(hour_format) + " " + end_cur.strftime(hour_format)
        try:
            sleep(delay + random.random())
            if verbose:
                print(f"downloading {timeframe} ... ", end="")
            pytrends = TrendReq(hl="fr-BE")
            pytrends.build_payload([topic_mid], geo=geo, timeframe=timeframe, cat=0)
            df = pytrends.interest_over_time()
            if df.empty:
                df = pd.DataFrame(
                    data={'date': pd.date_range(start=begin_cur, end=end_cur, freq='H'), topic_mid: 0}).set_index(
                    'date')
                df[batch_column] = -i
            else:
                df.drop(columns=['isPartial'], inplace=True)
                df[batch_column] = i
            i += 1
            df_tot = df_tot.append(df)
            df_tot.to_csv(file)
            if end_cur == end_tot:
                finished = True
            begin_cur += delta

            if end_cur + delta > end_tot:  # end of date
                end_cur = end_tot
                if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
                    begin_cur = end_cur - min_delta
            else:  # not end of date, increment
                end_cur = end_cur + delta

            if verbose:
                print("loaded")
            trials = 0
        except ResponseError as err:  # use a delay if an error has been received
            if str(err.response) == '<Response [500]>':
                write_file = f'../data/trends/collect/timeframe_not_available_{geo}.csv'
                f = open(write_file, "a+")
                f.writelines(f"{geo}, {topic_title}, {topic_mid}, {timeframe}\n")
                print(f"Error 500. Timeframe not available")
                if end_cur == end_tot:
                    finished = True
                begin_cur += delta

                if end_cur + delta > end_tot:  # end of date
                    end_cur = end_tot
                    if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
                        begin_cur = end_cur - min_delta
                else:  # not end of date, increment
                    end_cur = end_cur + delta
            else:
                trials += 1
                delay = 60
                if trials > 3:
                    write_file = f'../data/trends/collect/timeframe_not_available_{geo}.csv'
                    f = open(write_file, "a+")
                    f.writelines(f"{geo}, {topic_title}, {topic_mid}, {timeframe}\n")
                    print("ReadTimeOut. Timeframe not available")
                    trials = 0
                    if end_cur == end_tot:
                        finished = True
                    begin_cur += delta

                    if end_cur + delta > end_tot:  # end of date
                        end_cur = end_tot
                        if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
                            begin_cur = end_cur - min_delta
                    else:  # not end of date, increment
                        end_cur = end_cur + delta
                if verbose:
                    print(
                        f"Error when downloading (ResponseError). Retrying after sleeping during {delay} sec ... Trial : {trials}")
        except ReadTimeout:
            trials += 1
            delay = 60
            if trials > 3:
                write_file = f'../data/trends/collect/timeframe_not_available_{geo}.csv'
                f = open(write_file, "a+")
                f.writelines(f"{geo}, {topic_title}, {topic_mid}, {timeframe}\n")
                print("ReadTimeOut. Timeframe not available")
                trials = 0
                if end_cur == end_tot:
                    finished = True
                begin_cur += delta

                if end_cur + delta > end_tot:  # end of date
                    end_cur = end_tot
                    if end_cur - begin_cur < min_delta:  # must have a length of min 3 days
                        begin_cur = end_cur - min_delta
                else:  # not end of date, increment
                    end_cur = end_cur + delta
            if verbose:
                print(
                    f"Error when downloading (ReadTimeout). Retrying after sleeping during {delay} sec ... Trial : {trials}")
    return df_tot


def latest_trends_daily_date():
    """
    :return latest google trends date that can be queried using the daily requests
    """
    latest = date.today() - timedelta(days=3)
    return datetime(latest.year, latest.month, latest.day)


def dates_to_timeframe(start: date, stop: date) -> str:
    """
    Given two dates, returns a stringified version of the interval between
    the two dates which is used to retrieve data for a specific time frame
    from Google Trends.
    :param start: start date
    :param stop: stop date
    """
    return f"{start.strftime('%Y-%m-%d')} {stop.strftime('%Y-%m-%d')}"


def _fetch_data(pytrends, build_payload, timeframe: str) -> pd.DataFrame:
    """
    Attempts to fecth data and retries in case of a ResponseError.
    :param pytrends: object used for starting the TrendRequest on a localisation
    :param build_payload: object used for initializing a payload containing a particular word
    :param timeframe: string representing the timeframe
    :return a dataframe containing an interest over time for a particular topic
    """
    attempts = 0
    while True:
        try:
            build_payload(timeframe=timeframe)
            return pytrends.interest_over_time()
        except (ResponseError, ReadTimeout) as err:
            print(err)
            print(f'Trying again in {60 + 5 * attempts} seconds.')
            sleep(60 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attemps, abort fetching.')
                break
