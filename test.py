from src.covid_predictor.utils import *
from src.covid_predictor.DataGenerator import DataGenerator
import unittest
import pytest
from os import fspath


class TestDataGenerator(unittest.TestCase):
    
    def setUp(self) -> None:  # called before each test example
        pass
    
    @classmethod
    def setUpClass(cls) -> None:  # called once before running the tests
        date_begin = "2020-02-01"
        url_world = fspath(Path(__file__).parents[0]/"data/hospi/world.csv")
        url_pop = fspath(Path(__file__).parents[0]/"data/population.txt")
        population = get_world_population(url_pop)
        renaming = {v: k for k, v in european_geocodes.items()}
        geocodes = {k: v for k, v in european_geocodes.items() if population[k] > 1_000_000}
        df_hospi = hospi_world(url_world, geocodes, renaming, new_hosp=True, date_begin=date_begin)
        cls.scaler_generator = MinMaxScaler
        cls.df_hospi = df_hospi

    def test_normalisation_x(self):
        """
        test if X is normalised between 0,1 when using a MinMaxScaler(0, 1)
        """
        scaler_generator = self.scaler_generator
        df_hospi = self.df_hospi
        dg = DataGenerator(df_hospi, 20, 10, 'NEW_HOSP', scaler_generator=scaler_generator, scaler_type='batch')
        idx_begin = np.arange(100)
        X = dg.get_x(idx=idx_begin, scaled=True)
        self.assertEqual(1, X.max())
        self.assertEqual(0, X.min())

        idx_middle = np.arange(50, 120)
        X = dg.get_x(idx=idx_middle, scaled=True)
        self.assertEqual(1, X.max())
        self.assertEqual(0, X.min())

    def test_no_target(self):
        """
        test if a data generator can be created without target
        """
        scaler_generator = self.scaler_generator
        df_hospi = self.df_hospi
        n_forecast = 10
        dg_no_target = DataGenerator(df_hospi, 20, 0, 'NEW_HOSP', scaler_generator=scaler_generator, scaler_type='batch')
        batch_size_no_target = dg_no_target.batch_size
        dg_target = DataGenerator(df_hospi, 20, n_forecast, 'NEW_HOSP', scaler_generator=scaler_generator, scaler_type='batch')
        batch_size_target = dg_target.batch_size
        self.assertGreater(batch_size_no_target, batch_size_target)  # less values have been removed on a dg without target
        self.assertEqual(batch_size_no_target, batch_size_target + n_forecast)
        self.assertTrue(dg_no_target.no_target)
        self.assertFalse(dg_target.no_target)
        dg_no_target = DataGenerator(df_hospi, 20, n_forecast, '', scaler_generator=scaler_generator, scaler_type='batch')
        batch_size_no_target = dg_no_target.batch_size
        self.assertGreater(batch_size_no_target, batch_size_target)  # less values have been removed on a dg without target
        self.assertEqual(batch_size_no_target, batch_size_target + n_forecast)
        self.assertTrue(dg_no_target.no_target)


