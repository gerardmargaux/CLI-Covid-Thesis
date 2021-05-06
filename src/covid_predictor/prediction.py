from covid_predictor.utils import *
from covid_predictor.training import *
from covid_predictor.DataGenerator import DataGenerator
import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go


url_world = Path(__file__).parents[2]/"data/hospi/world.csv"
url_pop = Path(__file__).parents[2]/"data/population.txt"
url_trends = Path(__file__).parents[2]/"data/trends/model/"
url_hospi_belgium = Path(__file__).parents[2]/"data/hospi/be-covid-hospi.csv"
url_department_france = Path(__file__).parents[2]/"data/france_departements.csv"
url_hospi_france_new = Path(__file__).parents[2]/"data/hospi/fr-covid-hospi.csv"
url_hospi_france_tot = Path(__file__).parents[2]/"data/hospi/fr-covid-hospi-total.csv"


def prediction(model_name, model, df, parameters_dg, geo): 
    
    # Get parameters of the old generator
    n_samples=parameters_dg["n_samples"]
    n_forecast=parameters_dg["n_forecast"]
    target=parameters_dg["target"]
    scaler_gen=parameters_dg["scaler_generator"]
    europe=parameters_dg["europe"]
    augmented_pop=parameters_dg["augment_pop"]
    date_begin=parameters_dg["date_begin"]
    list_topics=parameters_dg["list_topics"]
    predict_one=parameters_dg["predict_one"]
    cumsum=parameters_dg["cumsum"]
    
    validation_metrics = [metric_get("MeanSquaredError"), metric_get('MeanAbsoluteError'), 
                      metric_get('RootMeanSquaredError')]
    
    if europe:
        population = get_world_population(url_pop)
        renaming = {v: k for k, v in european_geocodes.items()}
        geocodes = {k: v for k, v in european_geocodes.items() if population[k] > 1_000_000}
        df_hospi = hospi_world(url_world, geocodes, renaming, new_hosp=True, date_begin=date_begin)
        augment_population = {k: v/1000 for k, v in population.items()}
        
    else:
        geocodes = french_region_and_be
        population = pd.read_csv(url_department_france).groupby('regionTrends').agg({'population': 'sum'})
        augment_population = {k: pop['population'] / 100_000 for k, pop in population.iterrows()}  # pop per 100.000
        df_hospi = hospi_french_region_and_be(url_hospi_france_tot, url_hospi_france_new, url_hospi_belgium, 
                                            url_department_france, french_region_and_be, new_hosp=True, 
                                            tot_hosp=True, date_begin=date_begin)
        
    df_trends = create_df_trends(url_trends, list_topics, geocodes)  # Deal with augmented data
    for k in df_hospi.keys(): # Rolling average of 7 days 
        df_hospi[k] = df_hospi[k].rolling(7, center=True).mean().dropna()
        df_trends[k] = df_trends[k].rolling(7, center=True).mean().dropna()
    merged_df = {k: pd.merge(df_hospi[k], df_trends[k], left_index=True, right_index=True).dropna() for k,v in geocodes.items()}
    
    if scaler_gen == "MinMax":
        scaler_generator = MinMaxScaler
    else:
        scaler_generator = StandardScaler

    if model_name in ["encoder-decoder", "dense", "simple-auto-encoder"]:
        
        if europe:
            old_data_gen = DataGenerator(df, n_samples, n_forecast, target, scaler_generator=scaler_generator,
                        augment_merge=3, augment_adjacency=european_adjacency, augment_population=augment_population,
                        predict_one=predict_one, cumsum=cumsum, data_columns=None)
            
            new_data_gen = DataGenerator(merged_df, n_samples, 0, target=None, scaler_generator=scaler_generator,
                                augment_merge=3, augment_adjacency=european_adjacency, augment_population=augmented_pop,
                                predict_one=predict_one, cumsum=cumsum, data_columns=None)
        else:
            old_data_gen = DataGenerator(df, n_samples, n_forecast, target, scaler_generator=scaler_generator,
                        augment_merge=3, augment_adjacency=france_region_adjacency, augment_population=augment_population,
                        predict_one=predict_one, cumsum=cumsum, data_columns=None)
            
            new_data_gen = DataGenerator(merged_df, n_samples, 0, target=None, scaler_generator=scaler_generator,
                                augment_merge=3, augment_adjacency=france_region_adjacency, augment_population=augmented_pop,
                                predict_one=predict_one, cumsum=cumsum, data_columns=None)
        
        test_idx = [new_data_gen.batch_size-1]
        
        nb_datapoints = old_data_gen.batch_size
        train_idx = np.array(range(nb_datapoints))
        x_train = new_data_gen.get_x(train_idx, scaled=True)
        
        x_test = new_data_gen.get_x(test_idx, use_previous_scaler=True, geo=new_data_gen.loc_init)
        batch_input = len(x_test)
        
        y_train = old_data_gen.get_y(train_idx, scaled=True)
        y_pred = np.squeeze(model.predict(x_test, batch_size=batch_input))
        y_pred_real = old_data_gen.inverse_transform_y(y_pred, idx=[max(train_idx)], geo=new_data_gen.loc_init, return_type='dict_df')
        
    # Modiffication of the prediction format
    pred = y_pred_real[geo].reset_index().drop(columns=["LOC", "DATE"])
    range_dates = pd.date_range(start=datetime.today(), end=datetime.today() + timedelta(days=n_forecast-1))
    pred.columns = [date.strftime('%Y-%m-%d') for date in range_dates]
    final_pred = pred.transpose()
    final_pred.columns = ["Predictions"]
    final_pred.index.name = "DATE"
    
    # Modiffication of the hospi format
    final_hospi = df_hospi[geo].reset_index()
    to_drop = [elem for elem in list(final_hospi.columns) if elem not in [target, "DATE"]]
    final_hospi = final_hospi.drop(columns=to_drop).set_index("DATE")
    
    return final_pred, final_hospi


def prediction_reference(model, n_samples, n_forecast, target, geo, europe=False):
    models = {
        "baseline": prediction_baseline,
        "linear-regression": prediction_linear_regression
    }
    date_begin = "2020-02-01"
    
    if europe:
        df_hospi = hospi_world(url_world, geocodes, renaming, new_hosp=True, date_begin=date_begin)
        
    else:
        df_hospi = hospi_french_region_and_be(url_hospi_france_tot, url_hospi_france_new, url_hospi_belgium, 
                                            url_department_france, french_region_and_be, new_hosp=True, 
                                            tot_hosp=True, date_begin=date_begin)
    
    # Modiffication of the hospi format
    final_hospi = df_hospi[geo].reset_index()
    to_drop = [elem for elem in list(final_hospi.columns) if elem not in [target, "DATE"]]
    final_hospi = final_hospi.drop(columns=to_drop).set_index("DATE")
    
    y_train = final_hospi.tail(n_samples)[target].values
    prediction = models[model](y_train, n_forecast)
    
    final_pred = pd.DataFrame()
    range_dates = pd.date_range(start=datetime.today(), end=datetime.today() + timedelta(days=n_forecast-1))
    final_pred["DATE"] = [date.strftime('%Y-%m-%d') for date in range_dates]
    final_pred["Predictions"] = prediction
    final_pred = final_pred.set_index("DATE")
    
    return final_pred, final_hospi
    

def prediction_linear_regression(x_train, nb_test):
    axis = np.arange(len(x_train)).reshape(-1, 1)
    regr = LinearRegression().fit(axis, x_train)
    return regr.predict(np.arange(len(x_train), len(x_train) + nb_test).reshape(-1, 1))


def prediction_baseline(x_train, nb_test):
    return np.full(nb_test, x_train[-1])


def plot_prediction(pred_df, hospi, geo):
    color_train = '#1f77b4'
    color_prediction = '#ff7f0e'
    steps = np.pi / 30  # steps used between 2 points
    today = datetime.today().strftime('%Y-%m-%d')
    
    fig = plt.figure(figsize=(5, 4))
    hospi_df = hospi.tail(90)
    total_df = pd.concat([hospi_df, pred_df])
    total_df.columns = ["Value of the last days", "Predictions"]
    total_df = total_df.reset_index()
    pred_df = pred_df.reset_index()
    
    fig = px.scatter(total_df, x="DATE", y=total_df.columns, title=f"Prediction of the number of hospitalizations in {geo}")
    fig.data[0].update(mode='markers+lines')
    fig.for_each_trace(
        lambda trace: trace.update(marker_symbol="cross") if trace.name == "Predictions" else (),
    )
    fig.update_yaxes(title="Number of hospitalizations")
    fig.update_layout(legend_title="Type of points")
    fig.show()
