from pathlib import Path
import pandas as pd

from covid_predictor.utils import *


def actualize_trends(geocodes: Dict[str, str],
                     topics: Dict[str, str],
                     verbose: bool = True,
                     plot: bool = False,
                     only_hourly: bool = False,
                     refresh_daily: bool = True):
    """
    actualize the trends data in several steps:
    1) collect the hourly data
    2) correct the batch ids
    3) collect daily data on the gaps and form the final model data
    :param geocodes: dict of localizations
    :param topics: dict of topic_name: topic_code for trends
    :param verbose: whether to print information during the run or not
    :param plot: whether to plot the model data created or not
    :param only_hourly: whether to only actualize the hourly data or not
    :param refresh_daily: whether to refresh the daily data or not
    """
    # collect the hourly data
    for loc in geocodes:
        for topic_name, topic_code in topics.items():
            collect_historical_interest(topic_code, topic_name, loc, verbose=verbose)
    # correct the batch id
    if not only_hourly:
        for loc in geocodes:
            for topic_name, topic_code in topics.items():
                correct_batch_id(topic_name, topic_code, loc)
        # collect the data on the gaps
        collect_all_daily_gap(geocodes, topics, plot=plot, verbose=verbose, refresh=refresh_daily)


def actualize_trends_using_daily(geocodes: Dict[str, str],
                                 topics: Dict[str, str],
                                 verbose: bool = True,
                                 plot: bool = False,
                                 nb_mean: int = 10,
                                 overlap: int = 30,
                                 begin: datetime = None,
                                 refresh: bool = True):
    """
    actualize trends data using only daily requests and save it to the data/trends/model directory
    :param geocodes: dict of localizations
    :param topics: dict of topic_name: topic_code for trends
    :param verbose: whether to print information during the run or not
    :param plot: whether to plot the model data created or not
    :param overlap: number of days that must be overlapping
    :param nb_mean: number of mean requests to do to actualize the daily data
    :param refresh: whether to actually refresh the data or not. If False, generates model data based on the existing
        saved trends
    :param begin: first date where data is needed
    """
    max_query_days = 270
    if begin is None:
        begin = datetime.strptime("2020-01-01", "%Y-%m-%d")
    dir_daily = Path(__file__).parents[2]/"data/trends/collect_daily/"
    dir_model = Path(__file__).parents[2]/"data/trends/model/"
    plot_dir = Path(__file__).parents[2]/"plot/trends/"
    for loc in geocodes:
        for topic_name, topic_code in topics.items():
            filename = f"{dir_daily}{loc}-{topic_name}.csv"
            max_lag = np.floor(np.sqrt(nb_mean - 1))  # max lag used by dates iterator if every date can be queried
            length = max_query_days - 2 * max_lag  # length of the queries that will be done
            latest_day = latest_trends_daily_date()
            is_updated = False
            # load the saved csv file if it exist:
            if os.path.exists(filename):
                df_tot = pd.read_csv(filename, parse_dates=['date'], date_parser=date_parser_daily).set_index('date')
                # retrieve the dates already covered
                gb = df_tot.groupby("batch_id")
                df = gb.get_group(len(gb.size())-1)
                begin_covered = df.index.min().to_pydatetime()
                end_covered = df_tot.index.max().to_pydatetime()
                if end_covered == latest_day or not refresh:  # no need to update the file
                    is_updated = True
                else:
                    if (end_covered - begin_covered).days != length:  # the last batch registered was not the longest one
                        # remove the last batch as a longer can be queried
                        batch_id = df.iloc[0]["batch_id"]
                        df_tot = df_tot.loc[df_tot["batch_id"] != batch_id]
                        cur_begin = begin_covered
                        cur_end = cur_begin + timedelta(days=length)
                    else:  # the last batch was the best one that could be queried
                        batch_id = df_tot.iloc[-1]["batch_id"]
                        cur_begin = begin_covered + timedelta(days=(length - overlap))
                        cur_end = end_covered + timedelta(days=(length - overlap))
            else:  # no data existed previously
                if not refresh:  # cannot perform new queries if refresh == False
                    continue
                cur_begin = deepcopy(begin)
                cur_end = cur_begin + timedelta(days=length)
                df_tot = pd.DataFrame()
                batch_id = 0
            if not is_updated:
                list_dates = []
                last_length = max_query_days - nb_mean  # max length for the last query

                while cur_end < latest_day:
                    list_dates.append((cur_begin, cur_end))
                    cur_begin += timedelta(days=(length - overlap))
                    cur_end += timedelta(days=(length - overlap))

                cur_end = latest_day
                if (cur_end - cur_begin).days <= last_length:  # the last query can be done safely
                    list_dates.append((cur_begin, cur_end))
                else:  # need to split the last query into 2 sets of queries
                    cur_end = cur_begin + timedelta(last_length)
                    list_dates.append((cur_begin, cur_end))
                    cur_begin += timedelta(days=(length - overlap))
                    cur_end = latest_day
                    list_dates.append((cur_begin, cur_end))


                # send the queries
                for i, (date_from, date_to) in enumerate(list_dates):
                    if verbose:
                        print(f"{loc}-{topic_name}: retrieving batch {i+1}/{len(list_dates)} ")
                    df = mean_query(nb_mean, date_from, date_to, topic_code, loc, verbose=verbose)
                    df["batch_id"] = batch_id
                    df_tot = df_tot.append(df)
                    df_tot.to_csv(filename)
                    if verbose:
                        print(f"daily requests saved to {filename}")
                    batch_id += 1
            # construct the model data
            gb = df_tot.groupby("batch_id")
            list_df = [gb.get_group(x)[[topic_code]] for x in gb.groups]
            model = list_df[0]
            for df in list_df[1:]:
                model = merge_trends_batches(model, df, overlap + 1, topic_code, is_hour=False, verbose=False)
            model.to_csv(f"{dir_model}{loc}-{topic_name}.csv")
            if plot:
                plot_trends(model, topic_code, show=False)
                plt.title(f"{loc}: {topic_name}")
                plt.savefig(f"{plot_dir}{loc}-{topic_name}", facecolor='white')
                plt.show()


def actualize_hospi():
    url_hospi_belgium = "https://raw.githubusercontent.com/pschaus/covidbe-opendata/master/static/csv/be-covid-hospi.csv"
    url_hospi_france_new = "https://www.data.gouv.fr/fr/datasets/r/6fadff46-9efd-4c53-942a-54aca783c30c"
    url_hospi_france_tot = "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7"
    url_hospi_world = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    # hospi in belgium
    encoded_path_be = requests.get(url_hospi_belgium).content
    df_hospi_be = pd.read_csv(io.StringIO(encoded_path_be.decode("utf-8"))).drop(axis=1, columns='Unnamed: 0')
    df_hospi_be.to_csv(Path(__file__).parents[2]/'data/hospi/be-covid-hospi.csv', index=False)

    # hospi in France
    encoded_path_fr = requests.get(url_hospi_france_tot).content
    df_hospi_fr = pd.read_csv(io.StringIO(encoded_path_fr.decode("utf-8")))
    df_hospi_fr = df_hospi_fr.rename(columns=lambda s: s.replace('"', ''))
    for i, col in enumerate(df_hospi_fr.columns):
        df_hospi_fr.iloc[:, i] = df_hospi_fr.iloc[:, i].str.replace('"', '')
    df_hospi_fr.to_csv(Path(__file__).parents[2]/'data/hospi/fr-covid-hospi-total.csv', index=False)

    # new hospi in France
    encoded_path_fr_new = requests.get(url_hospi_france_new).content
    df_hospi_fr_new = pd.read_csv(io.StringIO(encoded_path_fr_new.decode("utf-8")))
    df_hospi_fr_new = df_hospi_fr_new.rename(columns=lambda s: s.replace('"', ''))
    for i, col in enumerate(df_hospi_fr_new.columns):
        df_hospi_fr_new.iloc[:, i] = df_hospi_fr_new.iloc[:, i].str.replace('"', '')
    df_hospi_fr_new.to_csv(Path(__file__).parents[2]/'data/hospi/fr-covid-hospi.csv', index=False)

    # hospi in the world
    encoded_path_world = requests.get(url_hospi_world).content
    df_world = pd.read_csv(io.StringIO(encoded_path_world.decode("utf-8")))
    df_world = df_world[['iso_code', 'location', 'date', 'icu_patients', 'hosp_patients', 'population']]
    df_world.drop(columns=['population']).to_csv(Path(__file__).parents[2]/'data/hospi/world.csv', index=False)

    # also write the world population
    pop = df_world.groupby("iso_code").agg("population").mean().to_dict()
    with open(Path(__file__).parents[2]/"data/population.txt", "w") as file:
        file.write(json.dumps(pop))
    return
