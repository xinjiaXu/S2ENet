import numpy as np


class ConfuseMatrixMeter:
    def __init__(self, n_class=2):
        self.n_class = n_class
        self.clear()

    def clear(self):
        self.confusion_matrix = np.zeros((self.n_class, self.n_class), dtype=np.float64)

    def update_cm(self, pr, gt):
        gt = np.asarray(gt).reshape(-1).astype(np.int64)
        pr = np.asarray(pr).reshape(-1).astype(np.int64)
        valid = (gt >= 0) & (gt < self.n_class) & (pr >= 0) & (pr < self.n_class)
        hist = np.bincount(self.n_class * gt[valid] + pr[valid], minlength=self.n_class ** 2)
        self.confusion_matrix += hist.reshape(self.n_class, self.n_class)
        scores = self.get_scores()
        return scores['F1']

    def get_scores(self):
        hist = self.confusion_matrix
        eps = np.finfo(np.float64).eps
        tn, fp, fn, tp = hist[0, 0], hist[0, 1], hist[1, 0], hist[1, 1]
        total = hist.sum()
        oa = (tp + tn) / (total + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (total * total + eps)
        kappa = (oa - expected) / (1 - expected + eps)
        return {'Kappa': kappa, 'IoU': iou, 'F1': f1, 'OA': oa,
                'recall': recall, 'precision': precision}
