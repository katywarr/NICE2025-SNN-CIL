import unittest
import numpy as np

from utils.one_hot_encoding import one_hot_encode_signal, one_hot_encode_signal_repeat, one_hot_signal_sliced


class TestOneHotEncoding(unittest.TestCase):

    def test_one_hot_encoding(self):
        signal = np.array([0, 5, 7, 5, 2])
        one_hot_signal = one_hot_encode_signal(signal=signal,
                                               phase_length=7)
        expected = np.array([
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, True, False, False],
                             [False, False, False, False, False, False, True],
                             [False, False, False, False, True, False, False],
                             [False, True, False, False, False, False, False]
                            ])

        self.assertTrue(np.array_equal(one_hot_signal, expected))

    def test_one_hot_encoding_repeat(self):
        signal = np.array([0, 5, 1, 5])
        one_hot_signal_repeated = one_hot_encode_signal_repeat(signal=signal,
                                                               phase_length=5,
                                                               repeats=2)
        expected = np.array([
                                [
                                    [False, False, False, False, False],
                                    [False, False, False, False, True],
                                    [True, False, False, False, False],
                                    [False, False, False, False, True],
                                ],
                                [
                                    [False, False, False, False, False],
                                    [False, False, False, False, True],
                                    [True, False, False, False, False],
                                    [False, False, False, False, True],
                                ]
                           ]
                          )
        self.assertTrue(np.array_equal(one_hot_signal_repeated, expected))

    def test_get_sliced_one_hot(self):
        signal = np.array([2, 0, 3, 4, 2])
        phase_length = 4
        one_hot_signal = one_hot_encode_signal(signal, phase_length)
        max_delay = 2

        one_hot_sliced = one_hot_signal_sliced(one_hot_signal, max_delay+1)

        expected = np.array([
            [[False, True, False],          # Time Step = 1
             [False, False, False],
             [False, False, True],
             [False, False, False],
             [False, True, False]],
            [[True, False, False],          # Time Step = 2
             [False, False, False],
             [False, True, False],
             [False, False, True],
             [True, False, False]],
            [[False, False, False],         # Time Step = 3
             [False, False, False],
             [True, False, False],
             [False, True, False],
             [False, False, False]],
            [[False, False, False],         # Time Step = 4
             [False, False, False],
             [False, False, False],
             [True, False, False],
             [False, False, False]]
                    ])

        self.assertTrue(np.array_equal(expected, one_hot_sliced))


if __name__ == '__main__':
    unittest.main()
