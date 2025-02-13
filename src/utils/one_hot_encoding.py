import numpy as np

def one_hot_encode_signal(signal: np.ndarray, phase_length: int) -> np.ndarray:
    """
    Return a one-hot version of the signal.
    This is useful for masking over a weight array.
    For example:

        Signal: [2 0 5]
        Phase length: 5

        One-hot_encoded
         [[False  True False False False]
          [False False False False False]
          [False False False False  True]]

    :param signal: np.ndarray
    :param phase_length: int
    Maximum value in a one_hot_signal. I.e. if one_hot_signal spans 16 values 0..15, this should be set to 15.
    :return: np.ndarray
    Repeated one-hot version of the signal with shape = (repeats, len(signal), phase length).
    """
    one_hot_signal = np.full(shape=(signal.size, phase_length + 1), fill_value=False)
    # TODO: During test, the signal is changed to float at higher levels. This was to allow
    # np.nan values. Might be a better solution than simply forcing to int here.
    one_hot_signal[np.arange(signal.size), signal.astype(int)] = True

    return np.array(one_hot_signal[:, 1:])


def one_hot_signal_sliced(one_hot_signal: np.ndarray, slice_size: int) -> np.ndarray:
    """
    Returns a one shot array of signal slices, where each slice represents the signal at a particular time step.
    :param one_hot_signal:
    :param slice_size:
    max delay + 1.
    :return:
    An array of shape (f, phase_length, max_delay+1)
    """
    one_hot_signal_sliced = np.full(shape=(one_hot_signal.shape[1], one_hot_signal.shape[0], slice_size),
                                    fill_value=False)
    for time_loc in range(one_hot_signal.shape[1] - 1, -1, -1):
        ts_slice_length = min(slice_size, one_hot_signal.shape[1] - time_loc)
        one_hot_signal_sliced[time_loc, :, 0:ts_slice_length] = one_hot_signal[:, time_loc:time_loc + ts_slice_length]
    return one_hot_signal_sliced

def one_hot_encode_signal_repeat(signal: np.ndarray, phase_length: int, repeats: int) -> np.ndarray:
    """
    Return a repeated one-hot version of the signal.
    This is useful for masking over a stacked set of weight arrays.
    For example:

        Signal: [2 0 5]
        Phase length: 5
        Repeats: 2

        One-hot_encoded
         [[False  True False False False]
         [False False False False False]
         [False False False False  True]]

        Repeated:
         [[[False  True False False False]
          [False False False False False]
          [False False False False  True]]

         [[False  True False False False]
          [False False False False False]
          [False False False False  True]]]

    :param signal: np.ndarray
    :param phase_length: int
    Maximum value in a one_hot_signal. I.e. if one_hot_signal spans 16 values 0..15, this should be set to 15.
    :param repeats: int
    Number of times the signal should be repeated
    :return: np.ndarray
    Repeated one-hot version of the signal with shape = (repeats, len(signal), phase length).
    """
    one_hot_signal_repeated = np.full(shape=(repeats, signal.size, phase_length), fill_value=False)
    one_hot_signal = one_hot_encode_signal(signal, phase_length)
    one_hot_signal_repeated[:] = one_hot_signal
    return np.array(one_hot_signal_repeated)


def one_hot_decode_value(one_hot_encoding: np.ndarray) -> int:
    """
    Returns the integer value represented by a one-hot signal array
    :param one_hot_encoding:
    :return:
    """
    locations = np.where(one_hot_encoding)[0]
    if len(locations) == 0:
        value = 0
    else:
        value = locations[0] + 1
    return value