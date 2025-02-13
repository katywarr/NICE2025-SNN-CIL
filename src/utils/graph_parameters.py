import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm

def graph_permanence(p_potent: float, p_dep: float, current_perm: float):
    print('Beginning at permanence: {}'.format(current_perm))
    permanence = np.array([current_perm])
    steps = np.array([0])
    while permanence[-1] < 1:
        steps = np.append(steps, steps[-1] + 1)
        new_permanence = permanence[-1] + (p_potent * permanence[-1])
        permanence = np.append(permanence, new_permanence)
    print('    Potentiation: {}. Took {} potentiation steps to reach threshold {}'
          .format(p_potent, steps[-1], permanence[-1]))
    steps_p = steps[-1]
    """
    plt.plot(steps, permanence)
    plt.show()
    plt.close()
    """
    permanence = np.array([current_perm])
    steps = np.array([0])
    while permanence[-1] > 0:
        steps = np.append(steps, steps[-1] + 1)
        new_permanence = permanence[-1] - (p_dep * (1-permanence[-1]))
        permanence = np.append(permanence, new_permanence)
    print('    Deprecation: {}. Took {} deprecation steps to reach threshold {}'
          .format(p_potent, steps[-1], permanence[-1]))
    steps_d = steps[-1]
    """
    plt.plot(steps, permanence)
    plt.show()
    plt.close()
    """
    return steps_p, steps_d

def graph_weights(w_potent, w_dep, steps_p, steps_d):
    steps_range = np.arange(steps_p)
    weights = np.array([1.0])
    for steps in steps_range:
        weights = np.append(weights, w_potent + weights[-1])
    steps_range = np.arange(steps_p+1)
    print('Spiking weight is {} following {} potentiation steps'.format(weights[-1], steps_p))
    """
    plt.plot(steps_range, weights)
    plt.show()
    plt.close()
    """
    steps_range = np.arange(steps_d)
    weights = np.array([1.0])
    steps_at_zero = None
    for steps in steps_range:
        weights = np.append(weights, weights[-1] - w_dep)
        if weights[-1] <= 0 and steps_at_zero is None:
            steps_at_zero = steps
    steps_range = np.arange(steps_d+1)
    print('Non-spiking weight is {} following {} deprecation steps'.format(weights[-1], steps_d))
    print('    Number of non-spikes to reach zero: {}'.format(steps_at_zero))
    """
    plt.plot(steps_range, weights)
    plt.show()
    plt.close()
    """

def steps_using_threshold(p_potent: float, p_dep: float, threshold_mean: float, threshold: float):
    permanence = np.array([0.5])
    steps = np.array([0])

    increment_ratio = threshold/threshold_mean

    while permanence[-1] < 1:
        steps = np.append(steps, steps[-1] + 1)
        new_permanence = permanence[-1] + (p_potent * increment_ratio)
        permanence = np.append(permanence, new_permanence)
    print('Potentiation: {}. Took {} potentiation steps to reach threshold {}'
          .format(p_potent, steps[-1], permanence[-1]))
    steps_p = steps[-1]

    permanence = np.array([0.5])
    steps = np.array([0])
    decrement_ratio = threshold_mean/threshold
    while permanence[-1] > 0:
        steps = np.append(steps, steps[-1] + 1)
        new_permanence = permanence[-1] - (p_dep * decrement_ratio)
        permanence = np.append(permanence, new_permanence)
    print('Deprecation: {}. Took {} deprecation steps to reach {}'
          .format(p_potent, steps[-1], permanence[-1]))
    steps_d = steps[-1]
    return steps_p, steps_d

def plot_functions(x:np.ndarray, mean, sd):
    #plt.plot(x, 1 / (1 + np.exp(-x)))
    # Normalise x
    #x_norm = (x-mean-sd)/(sd * 2 + 1)
    #plt.plot(x, x_norm)
    #plt.plot(x_norm, np.exp(x_norm))

    #plt.plot(x, x**3)
    #plt.plot(x, x**4)

    lowest = mean-sd
    norm_x = (x-lowest+1)/(sd*2+1)
    print(norm_x)
    plt.plot(x, norm_x)

    plt.plot(x, x**2)
    plt.plot(x, np.exp(norm_x))

    plt.show()
    plt.close()

def plot_trimmed_normal():
    sample = truncnorm.rvs(-4, 0, loc=45, scale=5, size=10000)
    _ = plt.hist(sample, bins=51, facecolor='g', edgecolor='k', alpha=0.4)

    x = np.linspace(-1, 1, 101)

    #plt.plot(x, norm.pdf(x) / (norm.cdf(1) - norm.cdf(-1)), 'k--', linewidth=1)
    plt.show()
    plt.close()

if __name__ == '__main__':
    p_potentiate = 0.2
    p_deprecate = 0.04
    w_potentiate = 0.4
    w_deprecate = 0.4
    threshold_mean = 30
    threshold_sd = 8
    neuron_threshold = 20

    #plot_trimmed_normal()
    steps_p, steps_d = graph_permanence(p_potent=p_potentiate, p_dep=p_deprecate, current_perm=0.5)
    #graph_weights(w_potent=w_potentiate, w_dep=w_deprecate, steps_p=steps_p, steps_d=steps_d)
    #steps_p, steps_d = steps_using_threshold(p_potent=p_potentiate, p_dep=p_deprecate,
                                             # threshold_mean=threshold_mean, threshold=neuron_threshold)


    #x = np.linspace(threshold_mean-threshold_sd, threshold_mean+threshold_sd, 10)
    #plot_functions(x, threshold_mean, threshold_sd)