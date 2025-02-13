import numpy as np


def get_voltages(weights: np.ndarray, one_hot_signal_repeat: np.ndarray, time_index: int) \
        -> np.ndarray:
    """
    Return the voltage vector for all the hidden neurons for a particular time step.
    :param weights: pre-synaptic weights for all hidden neurons.
        shape = (number of layer2 neurons, number of layer1 neurons)
    :param one_hot_signal_repeat: The one-hot encoding of the signal repeated over each of the pre-synaptic connections.
        shape = (number of layer2 neurons, number of layer1 neurons, one hot signal length), dtype=bool.
        where the one hot signal length is the signal_precision-1 (time window)
    :param time_index: Current time phase on the time window.
    """
    # Select the aspect of the signal that arrived at this timestep
    signal_slice_repeat = one_hot_signal_repeat[:, :, time_index]
    weights_masked_with_signal = np.ma.masked_array(weights, mask=~signal_slice_repeat)
    pre_synap_voltages = np.sum(weights_masked_with_signal, axis=1)
    return pre_synap_voltages


def get_spiking(f_h_weights: np.ndarray,
                one_hot_signal_repeat: np.ndarray,
                time_index: int,
                thresholds: np.ndarray) -> (np.ndarray, np.ndarray):
    """

    :return:
    """
    voltages = get_voltages(weights=f_h_weights, one_hot_signal_repeat=one_hot_signal_repeat, time_index=time_index)
    spiking_indices = np.where(np.logical_and(voltages >= thresholds, thresholds != -1))[0]
    spiking = np.full(shape=(f_h_weights.shape[0],), fill_value=False)
    spiking[spiking_indices] = True
    return spiking


