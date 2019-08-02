import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, f1_score, accuracy_score


def exact_binomial_conf(p, n):
    q = 1 - p
    return 1.96 * np.sqrt(p * q / n)


class PerformanceAnalysis():
    def __init__(self, X, Y, indet, pos_ind='above'):
        self.X = X
        self.Y = Y
        self.indet = indet
        self.pos_ind = pos_ind
        self.Xtrain = self.X[self.indet == 0]
        self.Ytrain = self.Y[self.indet == 0]
        self.Xtest = self.X[self.indet == 1]
        self.Ytest = self.Y[self.indet == 1]
        self.n_test = len(self.Ytest)

    def analyze_by_thresh(self, thresh):
        self.thresh = thresh
        if self.X.shape[1] == 1:
            y_pred = np.zeros(len(self.Ytest))
            if self.pos_ind == 'below':
                y_pred[self.Xtest[:, 0] < self.thresh] = 1
            else:
                y_pred[self.Xtest[:, 0] > self.thresh] = 1
        else:
            self.model, self.y_score = self.fit_model()
            y_pred = np.zeros(len(self.y_score))
            y_pred[self.y_score[:, 1] > self.thresh] = 1
        Ypos = self.Ytest[y_pred == 1]
        Yneg = self.Ytest[y_pred == 0]
        conf_mat = self.confusion_matrix(Ypos, Yneg)
        self.calculate_performance(**conf_mat)

    def sensitivity_analysis(self):
        f1_scores = []
        accs = []
        if self.X.shape[1] == 1:
            self.thresholds = np.arange(0, 100, 1)
            self.y_score = np.repeat(self.Xtest, 2, axis=1)
        else:
            _, _, self.thresholds = roc_curve(self.Ytest, self.y_score[:, 1])
        for thresh in self.thresholds:
            y_pred = np.zeros(len(self.y_score))
            if self.pos_ind == 'below':
                y_pred[self.y_score[:, 1] < thresh] = 1
            else:
                y_pred[self.y_score[:, 1] > thresh] = 1
            f1 = f1_score(self.Ytest, y_pred)
            f1_scores.append(f1)
            acc = accuracy_score(self.Ytest, y_pred)
            accs.append(acc)
        self.f1_scores = np.array(f1_scores)
        self.accs = np.array(accs)

    def plot_sensitivity(self, ax, label='Single-energy', color='darkorange', hu=True, xlim=(None, None)):
        ax.plot(self.thresholds, self.f1_scores, color=color, label=f'{label} F1-score')
        ax.plot(self.thresholds, self.accs, color=color, linestyle='--', label=f'{label} Accuracy')
        if hu:
            thresh_label = f'{self.thresh} HU'
            xlab = 'Threshold (HU)'
        else:
            thresh_label = self.thresh
            xlab = 'Threshold (model prediction)'
        ax.plot([self.thresh, self.thresh], [0, 1], 'k-', label=f'Threshold = {thresh_label}')
        ax.set_xlim(xlim)
        ax.set_ylim([0, 1])
        ax.set_xlabel(f'{xlab}', fontdict={'size': 14})
        ax.legend(loc='lower right', prop={'size': 10});

    def fit_model(self):
        model = LogisticRegression().fit(self.Xtrain, self.Ytrain)
        y_score = model.predict_proba(self.Xtest)
        return model, y_score

    def confusion_matrix(self, Ypos, Yneg):
        tp = np.zeros(len(Ypos))
        tp[Ypos == 1] = 1
        tp = tp.sum()
        fp = np.zeros(len(Ypos))
        fp[Ypos == 0] = 1
        fp = fp.sum()
        tn = np.zeros(len(Yneg))
        tn[Yneg == 0] = 1
        tn = tn.sum()
        fn = np.zeros(len(Yneg))
        fn[Yneg == 1] = 1
        fn = fn.sum()
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    def calculate_performance(self, tp, tn, fp, fn):
        sens = tp / (tp + fn)
        self.sens = {'mean': sens, 'ci': self._conf(sens)}
        spec = tn / (tn + fp)
        self.spec = {'mean': spec, 'ci': self._conf(spec)}
        ppv = tp / (tp + fp)
        self.ppv = {'mean': ppv, 'ci': self._conf(ppv)}
        npv = tn / (tn + fn)
        self.npv = {'mean': npv, 'ci': self._conf(npv)}
        acc = (tp + tn) / (tp + tn + fp + fn)
        self.acc = {'mean': acc, 'ci': self._conf(acc)}

    def print_performance(self, label):
        print(f"Performance measures for {label} model at threshold {self.thresh} HU:")
        print(f"Sensitivity: {self.sens['mean']:0.2f} +/- {self.sens['ci']:0.2f}")
        print(f"Specificity: {self.spec['mean']:0.2f} +/- {self.spec['ci']:0.2f}")
        print(f"PPV: {self.ppv['mean']:0.2f} +/- {self.ppv['ci']:0.2f}")
        print(f"NPV: {self.npv['mean']:0.2f} +/- {self.npv['ci']:0.2f}")
        print(f"Accuracy: {self.acc['mean']:0.2f} +/- {self.acc['ci']:0.2f}")
        print()

    def _conf(self, p):
        return exact_binomial_conf(p, self.n_test)
