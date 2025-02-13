import unittest
import os
import sklearn
import numpy as np
from utils.mnist_reader import read_openml_data, select_and_prepare_data, examples_per_label, set_data_precision

class MyTestCase(unittest.TestCase):

    def setUp(self):
        # Create a temporary folder to hold the data
        self.data_dir = os.getcwd()+os.sep+'temp_test_data'

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

    def tearDown(self):
        try:
            os.remove(os.path.join(self.data_dir+os.sep+'mnist_784', 'images.npy'))
            os.remove(os.path.join(self.data_dir+os.sep+'mnist_784', 'labels.npy'))
            os.remove(os.path.join(self.data_dir+os.sep+'fashion-MNIST', 'images.npy'))
            os.remove(os.path.join(self.data_dir+os.sep+'fashion-MNIST', 'labels.npy'))
        except FileNotFoundError:
            pass

    def test_read_raw_data_mnist(self):
        images, labels = read_openml_data(self.data_dir, dataset='mnist_784')
        self.assertEqual(images.shape, (70000, 784))
        self.assertEqual(labels.shape, (70000,))
        min_examples, unique_labels = examples_per_label(labels)
        self.assertEqual(min_examples, 6313)

    def test_read_raw_data_fashion(self):
        images, labels = read_openml_data(self.data_dir, dataset='fashion-MNIST')
        self.assertEqual(images.shape, (70000, 784))
        self.assertEqual(labels.shape, (70000,))
        min_examples, unique_labels = examples_per_label(labels)
        # This dataset is perfectly balanced
        self.assertEqual(min_examples, 7000)

    def test_read_raw_data_incorrect(self):
        self.assertRaises(sklearn.datasets._openml.OpenMLError,
                          lambda: read_openml_data(self.data_dir, dataset='incorrect'))

    def test_get_split_dataset(self):
        x = np.array([[0],[1],[2],[0],[1],[2],[1],[2],[0]])
        y = np.array(['0', '1', '2', '0', '1', '2', '1', '2', '0'])
        data_params = {'dataset': 'mnist_784',  # mnist_784
                       'trains_per_class': 5000,  # 5000,
                       'tests_per_class': 500,  # 1000,
                       'trains_in_test_set': False,
                       'training_labels': None,  # ['1', '8'], #None, # ['0', '2', '3', '4', '5', '6', '7', '8', '9'],
                       'testing_labels': None,
                       'precision_required': 7,
                       'shuffle': False,
                       'use_edge_detection': False}
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=1, tests_per_class=2)
        self.assertTrue(np.array_equal(x_train, np.array([[0],[1],[2]])))
        self.assertTrue(np.array_equal(y_train, np.array(['0','1','2'])))
        self.assertTrue(np.array_equal(x_test, np.array([[0],[0],[1],[1],[2],[2]])))
        self.assertTrue(np.array_equal(y_test, np.array(['0','0','1','1','2','2'])))


    def test_get_split_dataset_check_different(self):
        x = np.array([[0, 1],[1, 2],[2, 3],[0, 4],[1, 5],[2, 6],[1, 7],[2, 8],[0, 9]])
        y = np.array(['0', '1', '2', '0', '1', '2', '1', '2', '0'])
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=1, tests_per_class=2)
        self.assertTrue(np.array_equal(x_train, np.array([[0, 1],[1, 2],[2, 3]])))
        self.assertTrue(np.array_equal(y_train, np.array(['0','1','2'])))
        self.assertTrue(np.array_equal(x_test, np.array([[0, 4],[0, 9],[1, 5],[1, 7],[2, 6],[2, 8]])))
        self.assertTrue(np.array_equal(y_test, np.array(['0','0','1','1','2','2'])))
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=2, tests_per_class=1)
        self.assertTrue(np.array_equal(x_train, np.array([[0, 1],[0, 4],[1, 2],[1, 5],[2, 3],[2, 6]])))
        self.assertTrue(np.array_equal(y_train, np.array(['0','0','1','1','2','2'])))
        self.assertTrue(np.array_equal(x_test, np.array([[0, 9],[1, 7],[2, 8]])))
        self.assertTrue(np.array_equal(y_test, np.array(['0','1','2'])))

    def test_get_split_dataset_trains_in_test(self):
        x = np.array([[0, 1],[1, 2],[2, 3],[0, 4],[1, 5],[2, 6],[1, 7],[2, 8],[0, 9]])
        y = np.array(['0', '1', '2', '0', '1', '2', '1', '2', '0'])
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=1, tests_per_class=2,
                                                                   trains_in_test=True)
        self.assertTrue(np.array_equal(x_train, np.array([[0, 1],[1, 2],[2, 3]])))
        self.assertTrue(np.array_equal(y_train, np.array(['0','1','2'])))
        self.assertTrue(np.array_equal(x_test, np.array([[0, 1],[0, 4],[1, 2],[1, 5],[2, 3],[2, 6]])))
        self.assertTrue(np.array_equal(y_test, np.array(['0','0','1','1','2','2'])))
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=2, tests_per_class=1,
                                                                   trains_in_test=True)
        self.assertTrue(np.array_equal(x_train, np.array([[0, 1],[0, 4],[1, 2],[1, 5],[2, 3],[2, 6]])))
        self.assertTrue(np.array_equal(y_train, np.array(['0','0','1','1','2','2'])))
        self.assertTrue(np.array_equal(x_test, np.array([[0, 1],[1, 2],[2, 3]])))
        self.assertTrue(np.array_equal(y_test, np.array(['0','1','2'])))
        # If the tests are shared with the trains, there can be more of them without resulting in ValueError
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=2, tests_per_class=2,
                                                                   trains_in_test=True)
        self.assertTrue(np.array_equal(x_train, np.array([[0, 1],[0, 4],[1, 2],[1, 5],[2, 3],[2, 6]])))
        self.assertTrue(np.array_equal(y_train, np.array(['0','0','1','1','2','2'])))
        self.assertTrue(np.array_equal(x_test, np.array([[0, 1],[0, 4],[1, 2],[1, 5],[2, 3],[2, 6]])))
        self.assertTrue(np.array_equal(y_test, np.array(['0','0','1','1','2','2'])))

    def test_get_split_dataset_shuffle(self):
        x = np.array([[0],[1],[2],[0],[1],[2],[1],[2],[0]])
        y = np.array(['0', '1', '2', '0', '1', '2', '1', '2', '0'])
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=1, tests_per_class=2,
                                                                   shuffle_data=True)
        # This test only works because the random seed is constant
        self.assertTrue(np.array_equal(x_train, np.array([[2],[1],[0]])))
        self.assertTrue(np.array_equal(y_train, np.array(['2','1','0'])))
        self.assertTrue(np.array_equal(x_test, np.array([[2],[1],[0],[1],[0],[2]])))
        self.assertTrue(np.array_equal(y_test, np.array(['2','1','0','1','0','2'])))

    def test_get_split_dataset_fewer_training_labels(self):
        x = np.array([[0],[1],[2],[0],[1],[2],[1],[2],[0]])
        y = np.array(['0', '1', '2', '0', '1', '2', '1', '2', '0'])
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=1, tests_per_class=2,
                                                                   training_labels=['0', '1'])
        self.assertTrue(np.array_equal(x_train, np.array([[0],[1]])))
        self.assertTrue(np.array_equal(y_train, np.array(['0','1'])))
        self.assertTrue(np.array_equal(x_test, np.array([[0],[0],[1],[1],[2],[2]])))
        self.assertTrue(np.array_equal(y_test, np.array(['0','0','1','1','2','2'])))

    def test_get_split_dataset_fewer_test_labels(self):
        x = np.array([[0],[1],[2],[0],[1],[2],[1],[2],[0]])
        y = np.array(['0', '1', '2', '0', '1', '2', '1', '2', '0'])
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=1, tests_per_class=2,
                                                                   testing_labels=['0', '1'])
        self.assertTrue(np.array_equal(x_train, np.array([[0],[1],[2]])))
        self.assertTrue(np.array_equal(y_train, np.array(['0','1','2'])))
        self.assertTrue(np.array_equal(x_test, np.array([[0],[0],[1],[1]])))
        self.assertTrue(np.array_equal(y_test, np.array(['0','0','1','1'])))

    def test_get_split_dataset_fewer_testing_training_labels(self):
        x = np.array([[0],[1],[2],[0],[1],[2],[1],[2],[0]])
        y = np.array(['0', '1', '2', '0', '1', '2', '1', '2', '0'])
        x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class=1, tests_per_class=2,
                                                                   training_labels=['0', '1'],
                                                                   testing_labels=['1', '2'])
        self.assertTrue(np.array_equal(x_train, np.array([[0],[1]])))
        self.assertTrue(np.array_equal(y_train, np.array(['0','1'])))
        self.assertTrue(np.array_equal(x_test, np.array([[1],[1], [2], [2]])))
        self.assertTrue(np.array_equal(y_test, np.array(['1','1','2','2'])))

    def test_get_split_dataset_incorrect(self):
        x = np.array([[0],[1],[2],[0],[1],[2],[1],[2],[0]])
        y = np.array(['0', '1', '2', '0', '1', '2', '1', '2', '0'])
        # Insufficent data examples for 2 trains and 2 tests per class.
        self.assertRaises(ValueError, lambda: select_and_prepare_data(x, y, trains_per_class=2, tests_per_class=2))
        # Insufficent data examples for 2 trains and 4 tests per class when the trains are in the test set.
        self.assertRaises(ValueError, lambda: select_and_prepare_data(x, y, trains_per_class=2, tests_per_class=4,
                                                                      trains_in_test=True))

    def test_set_precision(self):
        orig_precision = 20
        x = np.arange(0, orig_precision)

        updated_precision = 4
        x_expected = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
        x_out = set_data_precision(x, updated_precision)
        self.assertTrue(np.array_equal(x_expected, x_out))

        updated_precision = 1
        x_expected = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        x_out = set_data_precision(x, updated_precision)
        self.assertTrue(np.array_equal(x_expected, x_out))

        updated_precision = 17
        x_expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17])
        x_out = set_data_precision(x, updated_precision)
        self.assertTrue(np.array_equal(x_expected, x_out))

    def test_set_data_precision_incorrect(self):
        x = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8]])
        self.assertRaises(ValueError, lambda: set_data_precision(x, required_precision=9))


if __name__ == '__main__':
    unittest.main()
