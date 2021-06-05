from covid_predictor.utils import *
from covid_predictor.training import *
from covid_predictor.DataGenerator import DataGenerator
import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go
from os import fspath


url_world = fspath(Path(__file__).parents[2]/"data/hospi/world.csv")
url_pop = fspath(Path(__file__).parents[2]/"data/population.txt")
url_trends = fspath(Path(__file__).parents[2]/"data/trends/model/")
url_hospi_belgium = fspath(Path(__file__).parents[2]/"data/hospi/be-covid-hospi.csv")
url_department_france = fspath(Path(__file__).parents[2]/"data/france_departements.csv")
url_hospi_france_new = fspath(Path(__file__).parents[2]/"data/hospi/fr-covid-hospi.csv")
url_hospi_france_tot = fspath(Path(__file__).parents[2]/"data/hospi/fr-covid-hospi-total.csv")

global n_forecast_final
global target_idx
global predict_one_final


def prediction(model_name, model, df, parameters_dg, geo, pretrained=True): 
    
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
    
    predict_one_final = predict_one
    n_forecast_final = n_forecast
    if pretrained:
        augment_merge = 0
    else:
        augment_merge = 3
    
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
        df_hospi[k] = df_hospi[k].rolling(7, center=True, min_periods=1).mean().dropna()
        df_trends[k] = df_trends[k].rolling(7, center=True, min_periods=1).mean().dropna()
    merged_df = {k: pd.merge(df_hospi[k], df_trends[k], left_index=True, right_index=True).dropna() for k,v in geocodes.items()}
    
    if scaler_gen == "MinMax":
        scaler_generator = MinMaxScaler
    else:
        scaler_generator = StandardScaler

    if model_name in ["encoder-decoder", "dense", "simple-auto-encoder"]:
        
        if europe:
            old_data_gen = DataGenerator(df, n_samples, n_forecast, target, scaler_generator=scaler_generator,
                        augment_merge=augment_merge, augment_adjacency=european_adjacency, augment_population=augment_population,
                        predict_one=predict_one, cumsum=cumsum, data_columns=None)
            
            new_data_gen = DataGenerator(merged_df, n_samples, 0, target=None, scaler_generator=scaler_generator,
                                augment_merge=augment_merge, augment_adjacency=european_adjacency, augment_population=augmented_pop,
                                predict_one=predict_one, cumsum=cumsum, data_columns=None)
        else:
            old_data_gen = DataGenerator(df, n_samples, n_forecast, target, scaler_generator=scaler_generator,
                        augment_merge=augment_merge, augment_adjacency=france_region_adjacency, augment_population=augment_population,
                        predict_one=predict_one, cumsum=cumsum, data_columns=None)
            
            new_data_gen = DataGenerator(merged_df, n_samples, 0, target=None, scaler_generator=scaler_generator,
                                augment_merge=augment_merge, augment_adjacency=france_region_adjacency, augment_population=augmented_pop,
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
    
    else:
        list_hosp_features = [
            'NEW_HOSP',
            'TOT_HOSP',
            #'TOT_HOSP_log',
            #'TOT_HOSP_pct',
        ]

        if europe:
            old_data_gen = DataGenerator(df, n_samples, n_forecast, target, scaler_generator=scaler_generator,
                        augment_merge=augment_merge, augment_adjacency=european_adjacency, augment_population=augment_population,
                        predict_one=predict_one, cumsum=cumsum, data_columns=list_hosp_features)
            
            new_data_gen = DataGenerator(merged_df, n_samples, n_forecast, target, scaler_generator=scaler_generator, scaler_type='batch',
                                augment_merge=augment_merge, augment_adjacency=european_adjacency, augment_population=augment_population,
                                predict_one=predict_one, cumsum=cumsum, data_columns=list_hosp_features)
        else:
            old_data_gen = DataGenerator(df, n_samples, n_forecast, target, scaler_generator=scaler_generator,
                        augment_merge=augment_merge, augment_adjacency=france_region_adjacency, augment_population=augment_population,
                        predict_one=predict_one, cumsum=cumsum, data_columns=list_hosp_features)
            
            new_data_gen = DataGenerator(merged_df, n_samples, n_forecast, target, scaler_generator=scaler_generator, scaler_type='batch',
                                augment_merge=augment_merge, augment_adjacency=france_region_adjacency, augment_population=augment_population,
                                predict_one=predict_one, cumsum=cumsum, data_columns=list_hosp_features)
            
        max_train = old_data_gen.batch_size
        train_idx = np.array(range(max_train))
        test_idx = [new_data_gen.batch_size-1]
        target_idx = new_data_gen.target_idx

        X_train_1 = new_data_gen.get_x(train_idx, scaled=False)
        Y_train = old_data_gen.get_y(train_idx, scaled=False)
        X_test_1 = new_data_gen.get_x(test_idx, scaled=False, geo=new_data_gen.loc_init)

        model_generator = get_baseline
        
        batch_size_train = len(X_train_1)
        batch_size_test = len(X_test_1)

        # First prediction based only on the hospitalizations
        first_model = model_generator(n_forecast=n_forecast, target_idx=target_idx, batch_input_shape=(batch_size_train, n_samples, old_data_gen.n_features))
        Y_test_pred_1 = first_model.predict(X_test_1)
        Y_train_pred_1 = first_model.predict(X_train_1)
        df_train_predicted_1 = old_data_gen.inverse_transform_y(Y_train_pred_1, idx=train_idx, return_type='dict_df', 
                                        inverse_tranform=False)
        df_test_predicted_1 = new_data_gen.inverse_transform_y(Y_test_pred_1, idx=test_idx, return_type='dict_df', 
                                                geo=new_data_gen.loc_init, inverse_tranform=False)
        df_train_1 = old_data_gen.inverse_transform_y(Y_train, idx=train_idx, return_type='dict_df',
                                                inverse_tranform=False)
        
        data_dg_c = [f'{topic}(t{i})' for i in range(-n_samples+1, 0, 1) for topic in list_topics] + [topic for topic in list_topics]
        target_df_c = [f'C(t+{i})' for i in range(1, n_forecast+1)]
        target_renaming = {new_data_gen.target_columns[i]: target_df_c[i] for i in range(n_forecast)}
        df_train_c = {loc : new_data_gen.df[loc].iloc[train_idx][data_dg_c] for loc in new_data_gen.df}
        threshold = 0.3
        threshold_fun = lambda x: [1+threshold if y >= 1+threshold else 1-threshold if y <= 1-threshold else y for y in x]
        for loc in df_train_c:
            df_train_c[loc][new_data_gen.target_columns] = df_train_1[loc] / df_train_predicted_1[loc]
            df_train_c[loc] = df_train_c[loc].rename(columns=target_renaming)
            df_train_c[loc] = df_train_c[loc].replace([-np.inf, np.inf, np.nan], 1)
            df_train_c[loc][target_df_c] = df_train_c[loc][target_df_c].apply(threshold_fun)
        
        df_train_c_old = pickle.load(open( Path(__file__).parents[2]/f"models/{model_name}_df_train_c_{target}.p", "rb" ))
            
        dg_2 = DataGenerator(df_train_c_old, n_samples, n_forecast, target='C', scaler_generator=scaler_generator, 
                          scaler_type='batch', augment_merge=0, predict_one=False, cumsum=False,
                          data_columns=[k for k in list_topics], no_lag=True)
        dg_2.set_loc_init(new_data_gen.loc_init)  # consider the other localisations as being augmented
        
        dg_2_new = DataGenerator(df_train_c, n_samples, n_forecast, target='C', scaler_generator=scaler_generator, 
                          scaler_type='batch', augment_merge=0, predict_one=False, cumsum=False,
                          data_columns=[k for k in list_topics], no_lag=True)
        dg_2_new.set_loc_init(new_data_gen.loc_init)  # consider the other localisations as being augmented
            
        X_train_2 = dg_2.get_x(scaled=True)
        C_train = dg_2.get_y(scaled=False)
        
        dg_2_new.set_scaler_values_x(dg_2.scaler_x)
        X_test_2 = dg_2_new.get_x(test_idx, scaled=True, use_previous_scaler=True, geo=dg_2_new.loc_init)
        batch_size_test = len(X_test_2)
        
        C_test_pred_2 = model.predict(X_test_2, batch_size=batch_size_test)
        df_test_c_predicted = dg_2_new.inverse_transform_y(C_test_pred_2, idx=test_idx, return_type='dict_df', inverse_tranform=False, geo=dg_2_new.loc_init)
        
        y_pred_real = {}
        inverse_columns = {j:i for i, j in parameters_dg["target_remaining"].items()}
        for loc in df_test_c_predicted:
            y_pred_real[loc] = df_test_predicted_1[loc] * df_test_c_predicted[loc].rename(columns=inverse_columns)


    # Modification of the prediction format
    pred = y_pred_real[geo].reset_index().drop(columns=["LOC", "DATE"])
    range_dates = pd.date_range(start=datetime.today(), end=datetime.today() + timedelta(days=n_forecast-1))
    pred.columns = [date.strftime('%Y-%m-%d') for date in range_dates]
    final_pred = pred.transpose()
    final_pred.columns = ["Predictions"]
    final_pred.index.name = "DATE"
    
    # Modification of the hospi format
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
        population = get_world_population(url_pop)
        renaming = {v: k for k, v in european_geocodes.items()}
        geocodes = {k: v for k, v in european_geocodes.items() if population[k] > 1_000_000}
        df_hospi = hospi_world(url_world, geocodes, renaming, new_hosp=True, date_begin=date_begin)
        
    else:
        df_hospi = hospi_french_region_and_be(url_hospi_france_tot, url_hospi_france_new, url_hospi_belgium, 
                                            url_department_france, french_region_and_be, new_hosp=True, 
                                            tot_hosp=True, date_begin=date_begin)
    
    for k in df_hospi.keys(): # Rolling average of 7 days 
        df_hospi[k] = df_hospi[k].rolling(7, center=True, min_periods=1).mean().dropna()
    
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
