from covid_predictor.DataGenerator import DataGenerator
from covid_predictor.utils import *
import pytest

    
def test_normalisation_x():
    """
    test if X is normalised between 0,1 when using a MinMaxScaler(0, 1)
    """
    date_begin = "2020-02-01"

    url_world = Path(__file__).parents[2]/"data/hospi/world.csv"
    url_pop = Path(__file__).parents[2]/"data/population.txt"
    population = get_world_population(url_pop)
    renaming = {v: k for k, v in european_geocodes.items()}
    geocodes = {k: v for k, v in european_geocodes.items() if population[k] > 1_000_000}
    df_hospi = hospi_world(url_world, geocodes, renaming, new_hosp=True, date_begin=date_begin)
    scaler_generator = MinMaxScaler
    dg = DataGenerator(df_hospi, 20, 10, 'NEW_HOSP', scaler_generator=scaler_generator, scaler_type='batch')
    idx_begin = np.arange(100)
    X = dg.get_x(idx=idx_begin, scaled=True)
    self.assertEqual(1, X.max())
    self.assertEqual(0, X.min())

    idx_middle = np.arange(50, 120)
    X = dg.get_x(idx=idx_middle, scaled=True)
    self.assertEqual(1, X.max())
    self.assertEqual(0, X.min())

