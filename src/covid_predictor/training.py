from pathlib import Path
import sklearn.metrics as metrics
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Reshape, TimeDistributed, LSTM, Lambda, Bidirectional, RepeatVector, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ProgbarLogger, History
from tensorflow.keras.metrics import get as metric_get
import itertools
import functools
import pickle

from covid_predictor.utils import *
from covid_predictor.DataGenerator import *


class MultiStepLastLayer(tf.keras.Model):
    """
    repeat the last hospitalisations given as input n_forecast time
    """
    def call(self, inputs, target_idx, predict_one, n_forecast, *args, **kwargs):
        a = inputs[:, -1:, target_idx:target_idx+1]  # target of the last days
        # a = tf.where(tf.not_equal(a, 0), tf.zeros_like(a), a)
        if not predict_one:
            return tf.tile(
                a,
                [1, n_forecast, 1]   # repeat target n_forecast time
            )
        else:
            return tf.tile(a, [1, 1, 1])
        
        
class MultiStepLastBaseline(tf.keras.Model):
    """
    repeat the last hospitalisations given as input n_forecast time
    """
    def __init__(self, n_forecast, target_idx, batch_input_shape=None, predict_one=False, *args, **kwargs):
        super(MultiStepLastBaseline, self).__init__(name='')
        self.total = tf.Variable(initial_value=tf.zeros((1,)), trainable=False)
        self.multi_step = MultiStepLastLayer()
        self.reshape = Reshape((n_forecast,))
        self.predict_one = predict_one
        self.target_idx = target_idx
        self.n_forecast = n_forecast
    
    def call(self, input_tensor, training=False):
        x = self.multi_step(input_tensor, self.target_idx, self.predict_one, self.n_forecast)
        return self.reshape(x)
        
    def get_weights(self):
        return None
    
    def set_weights(self, *args, **kwargs):
        return None
    

class LinearRegressionHospi(tf.keras.Model):
    """
    repeat the last hospitalisations given as input n_forecast time
    """
    def __init__(self, window_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        
    def predict(self, inputs, *args, **kwargs):
        y = inputs[:, -self.window_size:, target_idx]  # target of the last days
        length = len(inputs)
        x = np.arange(self.window_size).reshape(-1,1)  # dates of the target
        if not predict_one:
            result = np.zeros((length, n_forecast))
            for i in range(length):
                regr = LinearRegression().fit(x, y[i])  # linear regression of (days, target)
                result[i] = regr.predict(np.arange(self.window_size, self.window_size+n_forecast).reshape(-1,1))
        else:
            result = np.zeros((length, 1))
            for i in range(length):
                regr = LinearRegression().fit(x, y[i])
                result[i] = regr.predict([self.window_size+n_forecast-1])
        return result
    
    
class AssembleLayerTimeDist(tf.keras.Model):
    """
    assemble the prediction from the trends and the predictions from another model
    the same weight is used at every timestep (1 trainable parameter)
    """
    def __init__(self, input_shape):
        super(AssembleLayer, self).__init__(name='')
        self.dense_trends = tf.keras.layers.Dense(1, use_bias=False, 
                                                 kernel_constraint=tf.keras.constraints.MinMaxNorm(0.001, 0.2))
        batch_input_shape = (input_shape[0], input_shape[1], 1)
        self.time_dist = TimeDistributed(self.dense_trends, batch_input_shape=batch_input_shape)

    def call(self, input_tensor, training=False):
        x_trends = input_tensor[:, :, 1:]
        x_hosp = input_tensor[:, :, :1]
        x_trends = tf.keras.layers.Subtract()([x_trends, x_hosp])  # x_trends - x_hosp
        x_trends = self.time_dist(x_trends, training=training)  # apply simple weight
        return tf.keras.layers.Add()([x_trends, x_hosp])  # final prediction = x_hosp + (x_trends - x_hosp) * c


class MinMaxPositive(tf.keras.constraints.Constraint):
    
    def __init__(self, min_value, max_value):
        self.min_max_norm = tf.keras.constraints.MinMaxNorm(min_value, max_value)
        self.non_neg = tf.keras.constraints.NonNeg()
        
    def __call__(self, w):
        return self.min_max_norm(self.non_neg(w))
    
    
class AssembleLayer(tf.keras.Model):
    """
    assemble the prediction from the trends and the predictions from another model
    different weight are used at every timestep (n_forecast trainable parameter)
    """
    
    def __init__(self, batch_input_shape):
        super(AssembleLayer, self).__init__(name='')
        self.kernel = self.add_weight("kernel", shape=[1,batch_input_shape[1]], 
                                      constraint=MinMaxPositive(0.001, 0.2))

    def call(self, input_tensor, training=False):
        x_trends = input_tensor[:, :, 1]
        x_hosp = input_tensor[:, :, 0]
        x_trends = tf.keras.layers.Subtract()([x_trends, x_hosp])  # x_trends - x_hosp
        x_trends = tf.multiply(x_trends, self.kernel)  # apply simple weight: (x_trends - x_hosp) * c
        return tf.keras.layers.Add()([x_hosp, x_trends])  # final prediction = x_hosp + (x_trends - x_hosp) * c

    
def get_assemble(batch_input_shape):
    model = AssembleLayer(batch_input_shape)
    model.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])
    return model
    
        
def get_custom_linear_regression(window_size, *args, **kwargs):
    model = LinearRegressionHospi(window_size)
    model.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])
    return model


def get_baseline(*args, **kwargs):
    model = MultiStepLastBaseline(*args, **kwargs)
    model.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])
    return model


def get_encoder_decoder(batch_input_shape, n_forecast, predict_one=False):
    model = Sequential()
    model.add(LSTM(16, return_sequences=True, stateful=False, batch_input_shape=batch_input_shape, recurrent_dropout=0))
    model.add(LSTM(4, return_sequences=False, stateful=False))
    model.add(RepeatVector(n_forecast))  # repeat
    model.add(LSTM(4, return_sequences=True, stateful=False))  # dec
    if not predict_one:
        model.add(LSTM(16, return_sequences=True, stateful=False))  # dec
        model.add(TimeDistributed(Dense(1)))
        model.add(Reshape((n_forecast,)))
    else:
        model.add(LSTM(16, return_sequences=False, stateful=False))  # dec
        model.add(Dense(1))
        model.add(Reshape((1,)))
        
    def custom_loss_function(y_true, y_pred):
        weights_loss = np.array([(1/x) for x in range(1, n_forecast+1)])
        y_true = y_true * weights_loss
        y_pred = y_pred * weights_loss
        return tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    model.compile(loss=custom_loss_function, optimizer='adam', metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])
    return model


def get_simple_autoencoder(batch_input_shape, n_forecast, target_idx):
    model = Sequential()
    model.add(Lambda(lambda x: x[:,:,target_idx:target_idx+1], batch_input_shape=batch_input_shape))
    model.add(LSTM(32, return_sequences=False, stateful=False))
    model.add(Dense(n_forecast))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])
    return model


def get_dense_model(batch_input_shape, n_samples, n_forecast, target_idx, use_lambda=False):
    model = Sequential()
    if use_lambda:
        model.add(Lambda(lambda x: x[:,:,target_idx], batch_input_shape=batch_input_shape))  # select only the target of the previous days
        model.add(Dense(n_forecast))   # predict the next target based on the previous ones
    else:
        model.add(Dense(1, batch_input_shape=batch_input_shape))
        model.add(Reshape((n_samples,)))
        model.add(Dense(n_forecast))
    model.compile(loss=tf.losses.MeanSquaredError(),
                          metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_model(type_model, epochs, n_samples, n_forecast, target, date_begin, list_topics, cumsum, predict_one, europe, scaler_gen="MinMax"): 
    
    validation_metrics = [metric_get("MeanSquaredError"), metric_get('MeanAbsoluteError'), 
                      metric_get('RootMeanSquaredError')]

    url_world = Path(__file__).parents[2]/"data/hospi/world.csv"
    url_pop = Path(__file__).parents[2]/"data/population.txt"
    url_trends = Path(__file__).parents[2]/"data/trends/model/"
    url_hospi_belgium = Path(__file__).parents[2]/"data/hospi/be-covid-hospi.csv"
    url_department_france = Path(__file__).parents[2]/"data/france_departements.csv"
    url_hospi_france_new = Path(__file__).parents[2]/"data/hospi/fr-covid-hospi.csv"
    url_hospi_france_tot = Path(__file__).parents[2]/"data/hospi/fr-covid-hospi-total.csv"
    
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
    
    pickle.dump(merged_df, open(f"./models/{type_model}_merged_df_{target}.p", "wb" ))
    
    if scaler_gen == "MinMax":
        scaler_generator = MinMaxScaler
    else:
        scaler_generator = StandardScaler
        
    if type_model in ["encoder-decoder", "simple-auto-encoder", "dense"]:
        
        if europe:
            dg = DataGenerator(merged_df, n_samples, n_forecast, target, scaler_generator=scaler_generator,
                        augment_merge=3, augment_adjacency=european_adjacency, augment_population=augment_population,
                        predict_one=predict_one, cumsum=cumsum, data_columns=None)
        else:
            dg = DataGenerator(merged_df, n_samples, n_forecast, target, scaler_generator=scaler_generator,
                        augment_merge=3, augment_adjacency=france_region_adjacency, augment_population=augment_population,
                        predict_one=predict_one, cumsum=cumsum, data_columns=None)
        
        parameters_dg = {
            "n_samples": n_samples,
            "n_forecast": n_forecast,
            "target": target,
            "scaler_generator": scaler_gen,
            "europe": europe,
            "augment_pop": augment_population,
            "date_begin": date_begin,
            "list_topics": list_topics,
            "predict_one": predict_one,
            "cumsum": cumsum,
            "date": datetime.today().strftime('%d-%m-%Y')
        }
    
        n_features = dg.n_features
        target_idx = dg.target_idx
        nb_datapoints = dg.batch_size
    
        train_idx = np.array(range(nb_datapoints))
        x_train = dg.get_x(train_idx, scaled=True)
        y_train = dg.get_y(train_idx, scaled=True)
        
        batch_input = len(x_train)
        batch_input_shape=(batch_input, n_samples, n_features)
        
        models = {
            "encoder-decoder": get_encoder_decoder(batch_input_shape, n_forecast),
            "simple-auto-encoder": get_simple_autoencoder(batch_input_shape, n_forecast, target_idx),
            "dense": get_dense_model(batch_input_shape, n_samples, n_forecast, target_idx)
        }

        model = models[type_model]
        model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=batch_input)
    
    # Assembler model
    else:
        if scaler_gen == "MinMax":
            scaler_generator = MinMaxScaler
        else:
            scaler_generator = StandardScaler
        
        list_hosp_features = [
            'NEW_HOSP',
            'TOT_HOSP',
            #'TOT_HOSP_log',
            #'TOT_HOSP_pct',
        ]

        if europe:
            dg = DataGenerator(merged_df, n_samples, n_forecast, target, scaler_generator=scaler_generator, scaler_type='batch',
                                augment_merge=3, augment_adjacency=european_adjacency, augment_population=augment_population,
                                predict_one=predict_one, cumsum=cumsum, data_columns=list_hosp_features)
        else:
            dg = DataGenerator(merged_df, n_samples, n_forecast, target, scaler_generator=scaler_generator, scaler_type='batch',
                                augment_merge=3, augment_adjacency=france_region_adjacency, augment_population=augment_population,
                                predict_one=predict_one, cumsum=cumsum, data_columns=list_hosp_features)
        
        target_idx = dg.target_idx
        
        max_train = dg.batch_size
        train_idx = np.array(range(max_train))

        X_train_1 = dg.get_x(train_idx, scaled=False)
        Y_train = dg.get_y(train_idx, scaled=False)

        model_generator = get_baseline
        batch_size_train = len(X_train_1)

        # First prediction based only on the hospitalizations
        model = model_generator(n_forecast=n_forecast, target_idx=target_idx, batch_input_shape=(batch_size_train, n_samples, old_data_gen.n_features))
        Y_train_pred_1 = model.predict(X_train_1)
        df_train_predicted_1 = dg.inverse_transform_y(Y_train_pred_1, idx=train_idx, return_type='dict_df', 
                                                inverse_tranform=False)
        df_train_1 = dg.inverse_transform_y(Y_train, idx=train_idx, return_type='dict_df',
                                                inverse_tranform=False)  
        
        data_dg_c = [f'{topic}(t{i})' for i in range(-n_samples+1, 0, 1) for topic in list_topics] + [topic for topic in list_topics]
        target_df_c = [f'C(t+{i})' for i in range(1, n_forecast+1)]
        target_renaming = {dg.target_columns[i]: target_df_c[i] for i in range(n_forecast)}
        df_train_c = {loc : dg.df[loc].iloc[train_idx][data_dg_c] for loc in dg.df}
        threshold = 0.3
        threshold_fun = lambda x: [1+threshold if y >= 1+threshold else 1-threshold if y <= 1-threshold else y for y in x]
        for loc in df_train_c:
            df_train_c[loc][dg.target_columns] = df_train_1[loc] / df_train_predicted_1[loc]
            df_train_c[loc] = df_train_c[loc].rename(columns=target_renaming)
            df_train_c[loc] = df_train_c[loc].replace([-np.inf, np.inf, np.nan], 1)
            df_train_c[loc][target_df_c] = df_train_c[loc][target_df_c].apply(threshold_fun)
        
        pickle.dump(df_train_c, open(f"./models/{type_model}_df_train_c_{target}.p", "wb" ))
            
        dg_2 = DataGenerator(df_train_c, n_samples, n_forecast, target='C', scaler_generator=scaler_generator, 
                          scaler_type='batch', augment_merge=0, predict_one=False, cumsum=False,
                          data_columns=[k for k in list_topics], no_lag=True)
        dg_2.set_loc_init(dg.loc_init)  # consider the other localisations as being augmented
        
        parameters_dg = {
            "n_samples": n_samples,
            "n_forecast": n_forecast,
            "target": target,
            "scaler_generator": scaler_gen,
            "europe": europe,
            "augment_pop": augment_population,
            "date_begin": date_begin,
            "list_topics": list_topics,
            "predict_one": predict_one,
            "cumsum": cumsum,
            "date": datetime.today().strftime('%d-%m-%Y'),
            "target_remaining": target_renaming
        }
        
        X_train_2 = dg_2.get_x(scaled=True)
        C_train = dg_2.get_y(scaled=False)

        model_generator = get_dense_model

        batch_size_train = len(X_train_2)
        model = model_generator(batch_input_shape=(batch_size_train, n_samples, dg_2.n_features), n_samples=n_samples, n_forecast=n_forecast, target_idx=target_idx, use_lambda=False)
            
        model.fit(X_train_2, C_train, batch_size=batch_size_train, epochs=epochs, callbacks=None)

    
    return model, parameters_dg
