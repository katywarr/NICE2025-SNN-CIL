import unittest
from utils.similarity_scores import Scorer, SimilarityMetric
import numpy as np

class TestScorer(unittest.TestCase):

    def test_proportion_similarity(self):
        scorer = Scorer(metric=SimilarityMetric.PROPORTION)
        x1 = np.array([1,2,3,4,5,6])
        x2 = np.array([1,2,3,4,5,6])
        self.assertEqual(scorer.similarity(x1, x2), 1.0000)
        x2 = np.array([3,2,3,2,3,6])
        self.assertEqual(scorer.similarity(x1, x2), 0.5000)

    def test_best_score_proportion(self):
        scorer = Scorer(metric=SimilarityMetric.PROPORTION)
        x1 = np.array([1,1,1,1,1,1])
        x2 = np.array([[1,2,3,4,5,6],
                      [1,1,3,4,5,6],
                      [1,2,1,4,5,6],
                      [1,2,4,1,1,1],
                      [1,2,3,4,1,1]])
        best_score, index = scorer.get_closest(x1, x2)
        self.assertEqual(3, index)
        self.assertEqual(0.6667, best_score)

    def test_mse_similarity(self):
        scorer = Scorer(metric=SimilarityMetric.MSE)
        x1 = np.array([1,2,3,4,5,6])
        x2 = np.array([1,2,3,4,5,6])
        self.assertEqual(scorer.similarity(x1, x2), 0.0000)
        x2 = np.array([3,2,3,2,3,6])
        self.assertEqual(scorer.similarity(x1, x2), 2.0000)

    def test_best_score_mse(self):
        scorer = Scorer(metric=SimilarityMetric.MSE)
        x1 = np.array([1,1,1,1,1,1])
        x2 = np.array([[1,2,3,4,5,6],
                      [1,1,1,5,6,1],
                      [1,2,1,4,5,6],
                      [1,2,3,1,1,1],
                      [1,2,3,4,1,1]])
        best_score, index = scorer.get_closest(x1, x2)
        self.assertEqual(3, index)
        self.assertEqual(0.8333, best_score)

if __name__ == '__main__':
    unittest.main()
