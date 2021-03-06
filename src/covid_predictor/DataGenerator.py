from covid_predictor.utils import *

class DataGenerator:

    def __init__(self,
                 df: Dict[str, pd.DataFrame],
                 n_samples: int,
                 n_forecast: int,
                 target: str,
                 scaler_generator: callable,
                 scaler_type: str = "batch",
                 data_columns: List[str] = None,
                 no_scaling: List[str] = None,
                 cumsum: bool = False,
                 predict_one: bool = False,
                 augment_merge: int = 1,
                 augment_adjacency: List[Tuple[str, str]] = None,
                 augment_population: Dict[str, Union[int, float]] = None,
                 augment_feature_pop: List[str] = None,
                 no_lag: bool = False):
        """
        initialize a data generator. Takes a dict of {loc: dataframe} and use it to yield values suitable for training
        the data generator can augment the data by mixing regions together
        the values are padded so that each region contains the same number of datapoints. Only the right padding information is stored

        :param df: dataframe of values to use in order to generate X and Y. Must be double indexed by loc and date
        :param n_samples: number of timesteps in X
        :param n_forecast: number of timesteps in Y. If == 0, no target is set
        :param target: target to predict, will become the Y set. If == '', no target is set
        :param scaler_generator: generator of scaler to use
        :param scaler_type: one of "batch", "window", "whole"
        :param no_scaling: list of features that must not be scaled
        :param cumsum: if True, accumulates Y using cumsum
        :param predict_one: if True, the target is Y at time n_forecast. Otherwise, the target is Y in [t+1 ... t+n_forecast]
        :param augment_merge: number of regions to merge in order to augment the data. If <=1, no data augmentation is performed
        :param augment_adjacency: use the list of adjacent region to augment the data. If None, all regions
            will be mixed, even unadjacent
        :param augment_population: population of each region. Must not be None if augment_feature_pop contains values
        :param augment_feature_pop: list of features that should be weighted according to the population
        :param no_lag: if True, the dataframe will be considered as having the right format for training and add_lag
            will not be called on it
        """
        self.n_samples = n_samples
        self.n_forecast = n_forecast
        self.target = target
        self.cumsum = cumsum
        # transform the dict of dataframe into shape [samples, time_steps, features]
        # add lag to the data to be able to reshape correctly
        df = deepcopy(df)
        dummy_df = next(iter(df.values()))
        self.nb_loc = len(df)
        init_columns = list(dummy_df.columns)
        if data_columns is None:
            self.n_features = dummy_df.shape[1]
            data_columns = init_columns  # all columns are considered to be data columns
        else:
            self.n_features = len(data_columns)
        if target in data_columns:
            self.target_idx = data_columns.index(target)
            # TODO use option with cumsum
            if not cumsum:
                self.target_in_x = True
            else:
                self.target_in_x = False
        else:
            self.target_idx = None  # the target is not in the data columns, no need to specify it
            self.target_in_x = False

        # handle data generator without target
        self.no_target = target == '' or n_forecast == 0  # no target specified

        # pad the values: add 0 at the beginning and at the end
        smallest_dates = {}
        highest_dates = {}
        for k, v in df.items():  # get the dates covered on each loc
            dates = v.index.get_level_values('DATE')
            smallest_dates[k] = dates.min()
            highest_dates[k] = dates.max()
        min_date = min(smallest_dates.values())
        max_date = max(highest_dates.values())

        # add zeros (missing data) at the beginning and at the end, so that each df has the same number of values
        self.padded_idx = {}
        for k in df:
            if smallest_dates[k] > min_date:  # missing data at the beginning
                date_range = pd.date_range(min_date, smallest_dates[k] - timedelta(days=1))
                # loc_padded_idx = np.array(range(len(date_range)))
                nb_point = len(date_range)
                zeros = np.zeros(nb_point)
                pad_before = pd.DataFrame({**{'DATE': date_range, 'LOC': [k for _ in range(nb_point)]},
                                           **{col: zeros for col in init_columns}}).set_index(["LOC", "DATE"])
                df[k] = pad_before.append(df[k])
            else:
                pass
                # loc_padded_idx = np.array([])
            if highest_dates[k] < max_date:  # missing data at the end
                date_range = pd.date_range(highest_dates[k] + timedelta(days=1), max_date)
                #loc_padded_idx = np.append(loc_padded_idx, range(len(df[k]), len(df[k]) + len(date_range)))
                loc_padded_idx = np.arange(len(df[k]), len(df[k]) + len(date_range))
                nb_point = len(date_range)
                zeros = np.zeros(nb_point)
                pad_after = pd.DataFrame({**{'DATE': date_range, 'LOC': [k for _ in range(nb_point)]},
                                          **{col: zeros for col in init_columns}}).set_index(["LOC", "DATE"])
                df[k] = df[k].append(pad_after)
            else:
                loc_padded_idx = np.array([])
            self.padded_idx[k] = loc_padded_idx

        # augment the data
        self.loc_init = {k: k for k in df}
        self.loc_augmented = {}
        if augment_merge > 1:
            if augment_feature_pop is None:
                augment_feature_no_pop = init_columns
            else:
                augment_feature_no_pop = [col for col in init_columns if col not in augment_feature_pop]
                # filter to only take the columns already present in the dataframe
                augment_feature_pop = [i for i in augment_feature_pop if i in init_columns]
            for region_list in region_merge_iterator([loc for loc in df], augment_merge, augment_adjacency):
                region_code = '-'.join(sorted(set(region_list)))
                df[region_code] = sum(
                    [df[k][augment_feature_no_pop].reset_index().drop(columns=['LOC']).set_index('DATE') for k in
                     region_list])
                if augment_feature_pop:
                    sum_pop = sum([augment_population[k] for k in region_list])
                    df[region_code][augment_feature_pop] = sum(
                        [df[k][augment_feature_pop].reset_index().drop(columns=["LOC"]).
                        set_index('DATE') * augment_population[k] for k in region_list]) / sum_pop

                df[region_code]['LOC'] = region_code
                df[region_code] = df[region_code].reset_index().set_index(['LOC', 'DATE'])
                df[region_code] = df[region_code][init_columns]
                self.loc_augmented[region_code] = "Augmented region"

                if len(region_list) == 2:
                    self.padded_idx[region_code] = np.union1d(self.padded_idx[region_list[0]],
                                                              self.padded_idx[region_list[1]])
                else:
                    self.padded_idx[region_code] = reduce(np.union1d, ([self.padded_idx[k] for k in region_list]))
        # add data transformation
        add_transformations(df, data_columns)

        self.loc_all = {**self.loc_init, **self.loc_augmented}
        self.df_init = df  # contains the augmented data with padding and without lagged values
        self.padded_idx_init = deepcopy(self.padded_idx)  # padded indexes before the add lag
        # get the target and data columns
        self.predict_one = predict_one
        if predict_one:
            target_columns = [f"{target}(t+{n_forecast})"]
        else:
            target_columns = [f"{target}(t+{t})" for t in range(1, n_forecast + 1)]
        self.target_columns = target_columns
        self.data_columns_t0 = deepcopy(data_columns)
        # name of x columns across time
        self.data_columns = [f"{col}(t{t:+d})" for t in range(-n_samples + 1, 0) for col in
                             data_columns] + self.data_columns_t0

        # add lagged values
        if no_lag:  # the dataframe has already the right format
            self.df = df
            self.date_range = pd.date_range(min_date, max_date).to_pydatetime()
            days_removed = 0  # no day as been removed
        else:  # the dataframe must be constructed across time
            if self.no_target:  # no target specified
                self.df = {k: add_lag(v, - n_samples) for k, v in df.items()}
                self.date_range = pd.date_range(min_date + timedelta(days=(n_samples - 1)),
                                                max_date).to_pydatetime()
                days_removed = n_samples - 1
            else:  # a target exist
                self.df = {k: add_lag(v, - n_samples).join(add_lag(v[[target]], n_forecast),
                                                           how='inner') for k, v in df.items()}
                self.date_range = pd.date_range(min_date + timedelta(days=(n_samples - 1)),
                                                max_date - timedelta(days=n_forecast)).to_pydatetime()
                days_removed = n_forecast + n_samples - 1
        self.padded_idx = {k: (v - days_removed).astype(int) for k, v in self.padded_idx.items()}

        self.scaler_type = scaler_type  # can be "batch", "window", "whole"
        if no_scaling is None or target not in no_scaling:
            self.target_unscaled = False
        else:
            self.target_unscaled = True
        # unscaled columns names accross horizon
        if no_scaling is None:
            self.to_scale = list(range(self.n_features))
        else:
            no_scaling = [dummy_df.columns.get_loc(f"{name}(t-{n_samples})") for name in no_scaling]
            self.to_scale = [i for i in range(self.n_features) if i not in no_scaling]

        # construct the X and Y tensors
        self.X = []
        self.Y = []
        self.idx = {}
        idx = 0
        for i, (loc, val) in enumerate(self.df.items()):
            if not self.no_target:  # a target is specified
                y = val[target_columns].values
                if cumsum:
                    y = np.cumsum(y, axis=1)
                self.Y.append(y)
            x = val[self.data_columns].values
            x = x.reshape((len(x), n_samples, self.n_features))
            self.X.append(x)
            self.batch_size = len(x)
            self.idx[loc] = np.array(range(idx, idx + self.batch_size))
            idx += self.batch_size
            if i == 0:
                self.relative_idx = np.array(range(0, self.batch_size))
        self.X = np.concatenate(self.X)
        if self.no_target:
            self.Y = None
        else:
            self.Y = np.concatenate(self.Y)

        # self.X = np.concatenate([val.values.reshape((val.shape[0], n_samples, self.n_features))[:-n_forecast] for val in self.df.values()], axis=0)
        # self.Y = np.concatenate([val[target_columns].iloc[n_forecast:] for val in self.df.values()], axis=0)
        # self.Y = np.concatenate([val[target_columns].iloc[n_forecast:].cumsum(axis=1) for val in self.df.values()], axis=0)

        self.scaler_generator = None  # initialized by the set_scaler method
        self.set_scaler(scaler_generator)

    def set_scaler(self, scaler_generator: callable):
        """
        set the scaler used by the datagenerator
        :param scaler_generator: function that can be called to give the scaler to use
        """
        self.scaler_generator = scaler_generator
        # set the scaler
        # if window: Dict[str, Dict[int, [Dict[int, scaler]]]]: {loc: {feature_idx: {idx: scaler}}}
        # else: Dict[str, Dict[int, scaler]]: {loc: {feature_idx: scaler}}
        if self.scaler_type == "window":  # relative index
            self.scaler_x = {
                loc: {feature: {i: self.scaler_generator() for i in self.relative_idx} for feature in self.to_scale} for
                loc in self.idx}
            if self.no_target:
                self.scaler_y = None
            else:
                self.scaler_y = {loc: {i: self.scaler_generator() for i in self.relative_idx} for loc in self.idx}
        else:
            self.scaler_x = {loc: {feature: self.scaler_generator() for feature in self.to_scale} for loc in self.idx}
            if self.no_target:
                self.scaler_y = None
            else:
                self.scaler_y = {loc: self.scaler_generator() for loc in self.idx}

    def set_scaler_values_x(self, scaler_x: Dict):
        """
        copies the values of a scaler dict to this scaler
        """
        self.scaler_x = deepcopy(scaler_x)

    def set_loc_init(self, loc: Dict[str, str]):
        """
        set the initial loc and change the other loc to the augmented status
        :param loc: dict of localisation to consider as unaugmented localisations
        """
        self.loc_init = deepcopy(loc)
        self.loc_augmented = {k: v for k, v in self.loc_all.items() if k not in self.loc_init}

    def set_padded_idx(self, padded_idx):
        pass

    def get_x_dates(self, idx: Iterable = None):
        """
        gives the dates of the x values
        :param idx: index of dates of the x values to provide. If None, provide all possible dates
        :return array of dates, in one dimension
        """
        return self.date_range.to_pydatetime() if idx is None else self.date_range.to_pydatetime()[idx]

    def get_y_dates(self, idx: Iterable = None):
        """
        gives the dates of the y values
        :param idx: index of dates of the y values to provide. If None, provide all possible dates
        :return array of dates. If not self.predict_one, the array is 2d: [idx, n_forecast] else the array is 1d
        """
        if self.predict_one:
            dates = np.array([(self.date_range + timedelta(days=self.n_forecast)).to_pydatetime()])
        else:
            dates = np.column_stack(
                [(self.date_range + timedelta(days=i)).to_pydatetime() for i in range(1, self.n_forecast + 1)])
        return dates if idx is None else dates[idx]

    def get_x(self, idx: Iterable = None, geo: Dict[str, str] = None, scaled=True,
              use_previous_scaler: bool = False) -> np.array:
        """
        gives a X tensor, used to predict Y
        :param idx: index of the x values to provide. If None, provide the whole x values. Must be specified if
            scaler_type == 'batch'
        :param geo: localisations asked. If None, provide all loc
        :param scaled: if True, scale the data. Otherwhise, gives unscaled data
        :param use_previous_scaler: if True, use the scalers that were fit previously instead of new ones
        :return tensor of X values on the asked geo localisations and asked indexes.
        """
        if geo is None:
            geo = self.idx  # only the keys are needed
        if idx is not None:
            idx = np.array(idx)
        else:
            idx = self.relative_idx
        X = []
        for loc in geo:
            val = self.X[self.idx[loc], :]
            if not scaled:
                val = val[idx, :, :]
            else:  # need to scale x
                if self.scaler_type == "batch" or self.scaler_type == "whole":
                    if self.scaler_type == "batch":
                        val = val[idx, :, :]
                    # transform each feature
                    for feature_idx in self.to_scale:
                        if not use_previous_scaler:
                            if self.target_in_x and feature_idx == self.target_idx:
                                # need to add the values of y as well for the scaling
                                y_val = self.Y[self.idx[loc], :]
                                if self.scaler_type == "whole":
                                    y_val = y_val[-1, :]
                                else:
                                    y_val = y_val[idx[-1], :]
                            else:
                                y_val = []
                            old = val[:, -1, feature_idx].reshape(-1)  # get the values at t=0 on each window
                            new = val[0, :-1, feature_idx].reshape(-1)  # add the oldest values in the first window
                            new = np.append(new, y_val)
                            self.scaler_x[loc][feature_idx].fit(np.append(old, new).reshape((-1, 1)))  # fit the scaler
                        for t in range(self.n_samples):  # apply the transformation on the feature across time
                            val[:, t, feature_idx] = self.scaler_x[loc][feature_idx].transform(
                                val[:, t, feature_idx].reshape((-1, 1))).reshape((1, -1))
                    if self.scaler_type == "whole":
                        val = val[idx, :, :]
                elif self.scaler_type == "window":
                    if idx is None:
                        iterator = list(range(len(val)))
                    else:
                        iterator = idx
                    for feature_idx in self.to_scale:  # transform each feature
                        for i in iterator:
                            if use_previous_scaler:
                                val[i, :, feature_idx] = self.scaler_x[loc][feature_idx][i].transform(
                                    val[i, :, feature_idx].reshape((-1, 1))).reshape((1, -1))
                            else:
                                val[i, :, feature_idx] = self.scaler_x[loc][feature_idx][i].fit_transform(
                                    val[i, :, feature_idx].reshape((-1, 1))).reshape((1, -1))
                    val = val[idx, :, :]
            X.append(val)
        return np.concatenate(X)

    def get_y(self, idx: Iterable[int] = None, geo: Dict[str, str] = None, scaled: bool = True,
              use_previous_scaler: bool = False, repeated_values: bool = False) -> np.array:
        """
        gives a Y tensor, that should be predicted based on X
        :param idx: relative indexes of the y values to provide. If None, provide the whole y values
        :param geo: localisations asked. If None, provide all loc
        :param scaled: if True, scale the data. Otherwhise, gives unscaled data
        :param use_previous_scaler: if True, use the scalers that were fit previously instead of new ones
        :param repeated_values: should be False if for a loc y[i, t] == y[i+1, t-1] (This is the default behavior when
            no_lag == True in the constructor). If True, the scaler will be fit on the whole Y matrix
        :return tensor of y values on the asked geo localisations and asked indexes.
        """
        if geo is None:
            geo = self.idx  # only the keys are needed
        if idx is not None:
            idx = np.array(idx)
        else:
            idx = self.relative_idx
        Y = []
        for loc in geo:
            val = self.Y[self.idx[loc], :]
            if not scaled or self.target_unscaled:
                val = val[idx, :]
            else:  # need to scale y
                if self.scaler_type == "batch" or self.scaler_type == "whole":
                    if self.scaler_type == "batch":
                        val = val[idx, :]
                    # transform each feature
                    if self.predict_one:
                        if use_previous_scaler:
                            val = self.scaler_y[loc].transform(val)
                        else:
                            val = self.scaler_y[loc].fit_transform(val)
                    else:
                        if not use_previous_scaler:
                            if repeated_values or self.cumsum:
                                self.scaler_y[loc].fit(val.reshape((-1, 1)))  # fit the scaler
                            else:
                                if self.target_in_x:
                                    # need to add the values stored in x
                                    x_val = self.X[self.idx[loc], :, self.target_idx]
                                    if self.scaler_type == "whole":
                                        x_val = x_val[0, :]
                                    else:
                                        x_val = x_val[idx[0], :]
                                else:
                                    x_val = []
                                old = val[:, 0]  # get the values at t+1
                                old = np.append(old, x_val)
                                new = val[-1, 1:]  # add the most recent values at t+2 ... t+n_forecast
                                self.scaler_y[loc].fit(np.append(old, new).reshape((-1, 1)))  # fit the scaler
                        for t in range(val.shape[
                                           1]):  # apply the transformation on the target across time  # TODO optimize in one go
                            val[:, t] = self.scaler_y[loc].transform(val[:, t].reshape((-1, 1))).reshape((1, -1))
                    if self.scaler_type == "whole" and idx is not None:
                        val = val[idx, :]
                elif self.scaler_type == "window":
                    if idx is None:
                        iterator = range(len(val))
                    else:
                        iterator = idx
                    for i in iterator:
                        if use_previous_scaler:
                            val[i, :] = self.scaler_y[loc][i].transform(val[i, :].reshape((-1, 1))).reshape((1, -1))
                        else:
                            val[i, :] = self.scaler_y[loc][i].fit_transform(val[i, :].reshape((-1, 1))).reshape((1, -1))
                    val = val[idx, :]
            Y.append(val)
        return np.concatenate(Y)

    def inverse_transform_y(self, unscaled: np.array, geo: Union[str, Dict[str, str]] = None, idx: np.array = None,
                            return_type: str = 'array', inverse_transform: bool = True) -> Union[
        np.array, Dict[str, pd.DataFrame], Dict[str, np.array]]:
        """
        inverse transform the values provided, in order to get unscaled data
        uses the last scalers used in the get_y method
        :param unscaled: values to scale
        :param geo: localisation to scale. If None, all localisations are used. Must be in the order of Y:
            the first loc corresponds to the idx first entries in Y and so on
        :param idx: relative indexes of the scaling. Only relevant if the scaling type is batch
        :param return_type: can be one of "array", "dict_array" or "dict_df"
            - array: return a 2D np.array of values
            - dict_array: return a dict of {loc: np.array}
            - dict_df: return a dict of {loc: pd.DataFrame}
        :param inverse_transform: don't use any inverse transform
        """
        if geo is None:
            geo = self.idx  # only the keys are needed
        elif isinstance(geo, str):
            geo = {geo: self.idx[geo]}
        if idx is None or self.scaler_type != "batch":  # the scaler_type must be 'whole' or 'window'
            idx = self.relative_idx
            idx_dates = np.array(idx)
        else:
            idx_dates = np.array(idx)
            idx = np.array(range(len(idx)))

        val = np.zeros(unscaled.shape)
        return_df = {}
        if self.scaler_type == "batch" or self.scaler_type == "whole":
            offset = 0  # current offset in the Y tensor
            batch_size = len(idx)
            for loc in geo:
                loc_idx = idx + offset
                init_shape = unscaled[loc_idx, :].shape
                if inverse_transform:
                    val[loc_idx, :] = self.scaler_y[loc].inverse_transform(unscaled[loc_idx, :].reshape((-1, 1))).reshape(
                        init_shape)
                else:
                    val[loc_idx, :] = unscaled[loc_idx, :]
                offset += batch_size  # increment the offset to get the values from the next batch
                if return_type == 'dict_df':
                    dates_used = self.date_range[idx_dates]
                    multi_index = pd.MultiIndex.from_product([[loc], dates_used], names=['LOC', 'DATE'])
                    return_df[loc] = pd.DataFrame(val[loc_idx, :], columns=self.target_columns).set_index(
                        multi_index)
                elif return_type == 'dict_array':
                    return_df[loc] = val[loc_idx, :]
        elif self.scaler_type == "window":
            offset = 0  # current offset in the Y tensor
            batch_size = len(idx)
            if inverse_transform:
                for loc in geo:
                    for j, i in enumerate(idx):  # TODO implement inverse transform for window and corresponding return type
                        val[i + offset, :] = self.scaler_y[loc][i].inverse_transform(
                            unscaled[i + offset, :].reshape((-1, 1))).reshape((1, -1))
                    offset += batch_size  # increment the offset to get the values from the next batch
        return val if return_type == 'array' else return_df

    def loc_to_idx(self, loc):  # return absolute idx
        return self.idx[loc]

    def remove_padded_y(self, val: np.array, geo: Union[str, Dict[str, str]] = None, idx: np.array = None,
                        return_type: str = 'array') -> Union[np.array, Dict[str, pd.DataFrame], Dict[str, np.array]]:
        """
        remove the padded values of y and gives the result as a numpy array, or a dict of dataframe or numpy array
        :param val: values to scale
        :param geo: localisation to scale. If None, all localisations are used. Must be in the order of Y:
            the first loc corresponds to the idx first entries in Y and so on
        :param idx: relative indexes of the values. Only relevant if the scaling type is batch
        :param return_type: can be one of "array", "dict_array" or "dict_df"
            - array: return a 2D np.array of values
            - dict_array: return a dict of {loc: np.array}
            - dict_df: return a dict of {loc: pd.DataFrame}
        :return unpadded values of y. The type of return depends of the return_type parameter
        """
        implemented = ['array', 'dict_array', 'dict_df']
        if return_type not in implemented:
            raise Exception(f"return type should be one of {implemented}")

        if geo is None:
            geo = self.idx  # only the keys are needed
        elif isinstance(geo, dict):
            pass
        else:
            geo = {geo: self.idx[geo]}
        if idx is None:
            idx = self.relative_idx
        else:
            idx = np.array(idx)

        # remove the values where the data was padded
        filtered_values = {}
        batch_size = len(idx)
        offset = 0
        for loc in geo:
            # remove the indexes where the data was padded
            unpadded_idx = np.setdiff1d(idx, self.padded_idx[loc])
            dates_used = self.date_range[unpadded_idx]
            unpadded_idx = unpadded_idx + offset - idx[0]  # add offset inside matrix and remove first values if absent
            if return_type == 'dict_df':
                multi_index = pd.MultiIndex.from_product([[loc], dates_used], names=['LOC', 'DATE'])
                filtered_values[loc] = pd.DataFrame(val[unpadded_idx], columns=self.target_columns).set_index(
                    multi_index)
            else:
                filtered_values[loc] = val[unpadded_idx]
            offset += batch_size

        if return_type == 'dict_array' or return_type == 'dict_df':
            return filtered_values
        else:
            return np.concatenate([val for val in filtered_values.values()])

    def get_df_init(self) -> Dict[str, pd.DataFrame]:
        """
        return the initial dataframe that was used to construct the data, removing the padded values
        augmented values are included
        """
        init_df = {}
        for loc, df in self.df_init.items():
            unpadded_idx = np.setdiff1d(range(len(df)), self.padded_idx_init[loc])
            init_df[loc] = df.iloc[unpadded_idx]
        return init_df

    def __str__(self):
        """
        contains informations about
            - n_samples, n_forecast
            - data columns
            - target
            - scaling done
            - number of init and augmented regions, as well as their name
        """
        info = f'n_samples = {self.n_samples}, n_forecast = {self.n_forecast}\n'
        info += f'data = {self.data_columns_t0}\n'
        info += f'target = {self.target}\n'
        info += f'scaling = {self.scaler_generator}, scaling type = {self.scaler_type}\n'
        info += f'nb init regions = {len(self.loc_init)}, nb augmented regions = {len(self.loc_augmented)}\n'
        list_regions = [loc for loc in self.df]
        info += f'regions = {list_regions}'
        return info

    def time_idx(self, freq='M', format_date=False, boundary='inner') -> List[Tuple[np.array, Union[datetime, str]]]:
        """
        give the indexes corresponding to time interval

        :param freq: frequency for the time interval. supported:
            - 'M': monthly data
            - 'W': weekly data
            - 'D': daily data
        :param format_date: If True, transform the datetime into str, based on the frequency
        :param boundary: tell how to proceed for boundary: dates with values overlapping on multiple interval. supported:
            - 'inner': indices are split on dates where the n_forecast targets are in the next interval
            - 'outer': indices are split on dates where the target at t+n_forecast is in the next interval
            ex. with n_forecast = 2, freq='M':
                t       t+1   t+2
                29/01 | 30/01 31/01
                 -----------------     outer split
                30/01 | 31/01 01/02
                 -----------------     inner split
                31/01 | 01/02 02/02
        :return: tuples of (datetime, array of indices)
        """
        def round_dates(x: Tuple[int, datetime]) -> Tuple[int, datetime]:
            if boundary == 'inner':
                date_boundary = x[1] + timedelta(days=1)
            elif boundary == 'outer':
                date_boundary = x[1] + timedelta(days=self.n_forecast)
            else:
                raise ValueError(f'boundary is not a valid value. Found: {boundary}')

            if freq == 'M':
                begin_month = date_boundary.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
                rounded = x[0], begin_month
            elif freq == 'W':
                rounded = x[0], date_boundary - timedelta(days=date_boundary.weekday())
            elif freq == 'D':
                rounded = x
            else:
                raise ValueError(f'freq is not a valid value. Found: {freq}')
            return rounded

        def aggregate_dates(x, y) -> List[List[Union[List[int], datetime]]]:
            if isinstance(x, tuple):
                x = [[[x[0]], x[1]]]
            if x[-1][1] == y[1]:
                x[-1][0].append(y[0])
            else:
                x.append([[y[0]], y[1]])
            return x

        def to_np_array(x):
            for i in range(len(x)):
                x[i][0] = np.array(x[i][0])
                if format_date:
                    x[i][1] = datetime_to_str(x[i][1], freq)
            return x

        return to_np_array(reduce(aggregate_dates, map(round_dates, [(i, j) for i, j in enumerate(self.date_range)])))

    def walk_iterator(self, nb_test, periods_train=0, periods_eval=1, periods_test=1,
                      shift_test:int = 1, freq='M', boundary='inner'):
        """
        iterate over indexes, giving a split for training, evaluation and test set
        :param nb_test: total number of periods that must be evaluated in the test set
        :param periods_train: number of periods to use in training set. 0 = use all periods at each iteration
        :param periods_eval: number of periods to use in evaluation set
        :param periods_test: number of test periods included at each iteration. Default = 1 period per test set
        :param freq: frequency of the split
        :param boundary: tell how to proceed for boundary: dates with values overlapping on multiple interval. cf time_idx
            for details
        :return:
        """
        time_idx = self.time_idx(freq, format_date=True, boundary=boundary)
        nb_periods = len(time_idx)
        idx_test = max(nb_periods - nb_test * shift_test - periods_test + 1, periods_eval + periods_train)
        while idx_test + periods_test <= nb_periods:
            test_set = time_idx[idx_test:idx_test+periods_test]
            valid_set = time_idx[idx_test - periods_eval:idx_test]
            if periods_train == 0:
                training_set = time_idx[:idx_test - periods_eval]
            else:
                training_set = time_idx[idx_test - periods_eval - periods_train:idx_test - periods_eval]

            sets = [[training_set, periods_train], [valid_set, periods_eval], [test_set, periods_test]]
            for i in range(3):
                if sets[i][0]:
                    set_array = np.concatenate([sets[i][0][j][0] for j in range(len(sets[i][0]))])
                    if sets[i][1] > 1 or (i == 0 and sets[i][1] == 0):
                        sets[i] = set_array, f'{sets[i][0][0][1]} - {sets[i][0][-1][1]}'
                    else:
                        sets[i] = set_array, sets[i][0][-1][1]
                else:
                    sets[i] = [np.array([]), '']

            yield sets[0], sets[1], sets[2]
            idx_test += shift_test
