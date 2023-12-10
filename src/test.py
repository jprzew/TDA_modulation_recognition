import unittest
import pandas as pd
import numpy as np
from collections import namedtuple

from src.transform import add_and_drop

class TestDiagrams(unittest.TestCase):
    correct_file = 'data/stats_test_truth.pkl'
    tested_file = 'data/stats_test_transformed.pkl'

    def assertDiagramsEqual(self, diag1, diag2):
        assert len(diag1) == len(diag2)
        for array1, array2 in zip(diag1, diag2):
            np.testing.assert_array_equal(array1, array2)

    def assertDataframeEqual(self, a, b, msg):
        
        
        for column in a.columns:
            try: 
                if type(a[column].iloc[0]) == list:
                    for i in range(len(a[column])):
                        self.assertDiagramsEqual(a[column].iloc[i],
                                                 b[column].iloc[i])

                elif type(a[column].iloc[0]) == np.array:
                    for i in range(len(a[column])):
                        np.testing.assert_array_equal(a[column].iloc[i],
                                                      b[column].iloc[i])
                else:
                    pd.testing.assert_series_equal(a[column], b[column])


            except AssertionError as e:
                raise self.failureException(msg + f'(column: {column})') from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)
        
        Feature = namedtuple('Feature', 'name params')
        to_add = [Feature('point_cloud', {'dim': 10, 'step': 1}),
                  Feature('point_cloud', {'dim': 20, 'step': 1}),
                  Feature('point_cloud', {'dim': 2, 'step': 1}),
                  Feature('point_cloud', {'dim': 3, 'step': 1}),
                  Feature('point_cloud', {'dim': 4, 'step': 1}),
                  Feature('point_cloud', {'dim': 2, 'step': 'symbol_rate'}),
                  Feature('point_cloud', {'dim': 2, 'step': 30}),
                  Feature('point_cloud', {'dim': 4, 'step': 'symbol_rate'})]


        add_and_drop(input_file='data/stats_test_origin.pkl',
                     output_file=TestDiagrams.tested_file,
                     to_remove=[], to_add=to_add)
                  
        


    def test_columns(self):
        df_correct = pd.read_pickle(self.correct_file)
        df_tobetested = pd.read_pickle(self.tested_file)
        self.assertEqual(set(df_correct.columns), set(df_tobetested.columns),
                             'Dataframes have different columns')


    def test_data(self):
        df_correct = pd.read_pickle(self.correct_file)
        df_tobetested = pd.read_pickle(self.tested_file)
        self.assertEqual(df_correct, df_tobetested,
                         'Correct data is not equal to the tested data!')


if __name__ == '__main__':
    unittest.main()
