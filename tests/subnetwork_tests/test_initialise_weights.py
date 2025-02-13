import unittest
import numpy as np
from subnetwork.classifier_subnet import initialise_weights

class TestInitialiseWeights(unittest.TestCase):

    def test_initialise_weights_simple(self):
        source_population = 10
        target_population = 20
        sparsity = 0.1
        w_init=0.1
        proportion=True

        h_learning_weights = initialise_weights(source_population_size=source_population,
                                                target_population_size=target_population,
                                                sparsity=sparsity,
                                                w_init=w_init,
                                                proportion=proportion)
        self.assertEqual(target_population, h_learning_weights.shape[0])
        self.assertEqual(source_population, h_learning_weights.shape[1])
        total_weights = source_population * target_population
        connections = total_weights * sparsity
        self.assertEqual(connections, np.count_nonzero(h_learning_weights))
        self.assertEqual(connections, np.count_nonzero(h_learning_weights==w_init))


if __name__ == '__main__':
    unittest.main()
