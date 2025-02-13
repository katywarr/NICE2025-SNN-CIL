import unittest

from utils.one_hot_encoding import one_hot_encode_signal_repeat
from subnetwork.neuron_utils import get_spiking
import numpy as np


"""
The following verification functions are used to validate the behaviour of those in neuron_utils.
Unlike the neuron_utils functions, they do not loop through all the neurons but act upon the numpy arrays.
Empirical testing shows this is faster when the numpy arrays are small (i.e. the neurons in the learning pool
and mature hidden layer) are fewer, but is less efficient when the numpy arrays grow - hence these functions
are not in the main code.
"""
def get_voltages_verify(weights_repeated: np.ndarray,
                        one_hot_signal_repeat: np.ndarray) \
        -> np.ndarray:
    """
    Return the voltage vector for all the hidden neurons for all the timesteps to prevent looping
    through all the timesteps.
    """
    weights_masked_with_signal = np.ma.masked_array(weights_repeated, mask=~one_hot_signal_repeat)
    pre_synap_voltages = np.sum(weights_masked_with_signal, axis=1)
    return pre_synap_voltages


def get_spiking_verify(weights_repeated: np.ndarray,
                       one_hot_signal_repeat: np.ndarray,
                       thresholds_repeated: np.ndarray) -> (np.ndarray, np.ndarray):
    """

    :return:
    """
    voltages = get_voltages_verify(weights_repeated=weights_repeated,
                                   one_hot_signal_repeat=one_hot_signal_repeat)
    spiking_indices = np.where(np.logical_and(voltages >= thresholds_repeated,
                                              thresholds_repeated != -1))[0]
    # Return the spiking array - one boolean for each hidden neuron.
    spiking = np.full(shape=(weights_repeated.shape[0],), fill_value=False)
    spiking[spiking_indices] = True
    return spiking


def compare_predict_functions(f_h_weights: np.ndarray,
                              phase_length: int,
                              one_hot_signal_repeat: np.ndarray,
                              thresholds: np.ndarray):
    # Non-optimised predict with loop
    prediction = np.full(shape=(f_h_weights.shape[0],), fill_value=False)
    for time_index in range(phase_length - 1, -1, -1):
        spiking_this_time = get_spiking(f_h_weights=f_h_weights,
                                        one_hot_signal_repeat=one_hot_signal_repeat,
                                        time_index=time_index,
                                        thresholds=thresholds)
        prediction[spiking_this_time] = True

    # Optimised prediction without loop.
    # Peparing the repeated arrays can occur once for multiple predicts.
    weights_repeated = np.repeat(f_h_weights[:, :, np.newaxis],
                                 one_hot_signal_repeat.shape[2], axis=2)
    thresholds_repeated = np.repeat(thresholds[:, np.newaxis],
                                    one_hot_signal_repeat.shape[2], axis=1)
    prediction_fast = get_spiking_verify(weights_repeated=weights_repeated,
                                     one_hot_signal_repeat=one_hot_signal_repeat,
                                     thresholds_repeated=thresholds_repeated)
    return prediction, prediction_fast

class MyTestCase(unittest.TestCase):

    def test_get_voltages_for_tim_loc(self):
        h = 3
        phase_length = 7
        signal =       np.array([2, 2, 5, 2, 5, 7, 0, 2])
        threshold = 3
        f_h_weights = np.array([[1, 0, 0, 1, 0, 0, 0, 0],
                                [1, 1, 2, 0, 1, 1, 1, 1],
                                [0, 0, 1, 0, 1, 0, 1, 0]])
        print(f_h_weights.shape)

        weight_scale= np.array([2, 1, 1])
        thresholds = threshold/weight_scale
        one_hot_signal_repeat = one_hot_encode_signal_repeat(signal, repeats=h, phase_length=phase_length)

        spiking = get_spiking(f_h_weights=f_h_weights,
                              one_hot_signal_repeat=one_hot_signal_repeat,
                              thresholds=thresholds,
                              time_index=7-1
                              )
        spiking_indices = np.where(spiking)[0]
        self.assertTrue(np.array_equal(np.array([]),  spiking_indices))

        spiking = get_spiking(f_h_weights=f_h_weights,
                              one_hot_signal_repeat=one_hot_signal_repeat,
                              thresholds=thresholds,
                              time_index=6-1)
        spiking_indices = np.where(spiking)[0]
        self.assertTrue(np.array_equal(np.array([]),  spiking_indices))

        spiking = get_spiking(f_h_weights=f_h_weights,
                              one_hot_signal_repeat=one_hot_signal_repeat,
                              thresholds=thresholds,
                              time_index=5-1)
        spiking_indices = np.where(spiking)[0]
        self.assertTrue(np.array_equal(np.array([1]), spiking_indices))

        spiking = get_spiking(f_h_weights=f_h_weights,
                              one_hot_signal_repeat=one_hot_signal_repeat,
                              thresholds=thresholds,
                              time_index=4-1)
        spiking_indices = np.where(spiking)[0]
        self.assertTrue(np.array_equal(np.array([]),  spiking_indices))

        spiking = get_spiking(f_h_weights=f_h_weights, one_hot_signal_repeat=one_hot_signal_repeat,
                              thresholds=thresholds,
                              time_index=3-1)
        spiking_indices = np.where(spiking)[0]
        self.assertTrue(np.array_equal(np.array([]), spiking_indices))

        spiking = get_spiking(f_h_weights=f_h_weights, one_hot_signal_repeat=one_hot_signal_repeat,
                              thresholds=thresholds,
                              time_index=2-1)
        spiking_indices = np.where(spiking)[0]
        self.assertTrue(np.array_equal(np.array([0, 1]),  spiking_indices))

    def test_get_voltages_for_tim_loc_comp(self):
        h = 3
        phase_length = 7
        signal =       np.array([2, 2, 5, 2, 5, 7, 0, 2])
        f_h_weights = np.array([[1, 0, 0, 1, 0, 0, 0, 0],
                                [1, 1, 2, 0, 1, 1, 1, 1],
                                [0, 0, 1, 0, 1, 0, 1, 0]])
        print(f_h_weights.shape)

        thresholds = np.array([-1, -1, 2])
        print(thresholds)
        one_hot_signal_repeat = one_hot_encode_signal_repeat(signal, repeats=h, phase_length=phase_length)

        spiking = get_spiking(f_h_weights=f_h_weights, one_hot_signal_repeat=one_hot_signal_repeat,
                              thresholds = thresholds,
                              time_index=7-1)
        spiking_indices = np.where(spiking)[0]
        self.assertTrue(np.array_equal(np.array([]),  spiking_indices))

    def test_optimised_predict(self):
        h = 3
        signal = np.array([2, 2, 5, 2, 5, 7, 0, 2])
        phase_length = np.max(signal)
        f_h_weights = np.array([[1, 0, 0, 1, 0, 0, 0, 1],
                                [1, 1, 1, 0, 1, 1, 1, 1],
                                [0, 0, 1, 0, 1, 0, 1, 0]])
        one_hot_signal_repeat = one_hot_encode_signal_repeat(signal, repeats=h, phase_length=phase_length)
        thresholds = np.array([2, 2, 3])

        predict, predict_fast = compare_predict_functions(
            f_h_weights=f_h_weights,
            phase_length=phase_length,
            one_hot_signal_repeat=one_hot_signal_repeat,
            thresholds=thresholds)
        self.assertTrue(np.array_equal(predict, predict_fast))

    def test_optimised_predict_2(self):
        h = 3
        signal =       np.array([2, 2, 1, 1, 1, 3, 0, 0])
        f_h_weights = np.array([[1, 0, 0, 1, 0, 0, 0, 1],   # * 0
                                [0, 0, 0, 0, 0, 1, 0, 0],   # * 3
                                [0, 0, 1, 0, 1, 0, 1, 0]])  # * 2
        phase_length = np.max(signal)

        one_hot_signal_repeat = one_hot_encode_signal_repeat(signal, repeats=h, phase_length=phase_length)
        thresholds = np.array([3, 1, 3])
        predict, predict_fast = compare_predict_functions(
            f_h_weights=f_h_weights,
            phase_length=phase_length,
            one_hot_signal_repeat=one_hot_signal_repeat,
            thresholds=thresholds)

        self.assertTrue(np.array_equal(predict, predict_fast))

    def test_optimised_predict_with_varying_weights(self):
        h = 3
        signal =       np.array([2, 2, 1, 1, 1, 3, 0, 0])
        f_h_weights = np.array([[1, 0, 0, 2, 0, 0, 0, 1],   # * 0
                                [0, 0, 0, 0, 0, 0.2, 0, 0],   # * 3
                                [0, 0, 1, 0, 1, 0, 1, 0]])  # * 2
        phase_length = np.max(signal)

        one_hot_signal_repeat = one_hot_encode_signal_repeat(signal, repeats=h, phase_length=phase_length)
        thresholds = np.array([2, 2, 3])
        predict, predict_fast = compare_predict_functions(
            f_h_weights=f_h_weights,
            phase_length=phase_length,
            one_hot_signal_repeat=one_hot_signal_repeat,
            thresholds=thresholds)

        self.assertTrue(np.array_equal(predict, predict_fast))

if __name__ == '__main__':
    unittest.main()
