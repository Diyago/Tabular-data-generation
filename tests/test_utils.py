import unittest
import pandas as pd

from tabgan.utils import make_two_digit, get_year_mnth_dt_from_date


class TestUtils(unittest.TestCase):
    def test_make_two_digit(self):
        self.assertEqual(make_two_digit('1'), '01')
        self.assertEqual(make_two_digit('12'), '12')
        self.assertEqual(make_two_digit('123'), '123')


class TestUtils(unittest.TestCase):
    def test_get_year_mnth_dt_from_date(self):
        # create a sample dataframe
        df = pd.DataFrame({
            'Date': ['2022-01-01', '2022-02-01', '2022-03-01']
        })
        # call the function
        result = get_year_mnth_dt_from_date(df)
        # check if the output is correct
        self.assertEqual(list(result.columns), ['Date', 'year', 'month', 'day'])
        self.assertEqual(list(result['year']), [2022, 2022, 2022])
        self.assertEqual(list(result['month']), [1, 2, 3])
        self.assertEqual(list(result['day']), [1, 1, 1])
