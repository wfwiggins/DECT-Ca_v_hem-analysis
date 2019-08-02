import numpy as np
from scipy import interp
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score


class QuantAnalysis():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def analyze(self):
        if self.X.shape[1] == 1:
            self.thresh_hus = []
        else:
            self.threshs = []
        tprs = []
        self.base_fpr = np.linspace(0, 1, 101)
        self.idx = np.arange(0, len(self.Y))
        self.ct = 0
        np.random.seed(42)
        rkf = RepeatedKFold(n_splits=5, n_repeats=100)
        for train, test in rkf.split(self.Y):
            model, y_score, fpr, tpr, thresholds = self.fit_roc(train, test)
            tprs.append(tpr)
            self.ct += 1
            f1_scores, accs = self.sensitivity_analysis(test, y_score, thresholds)
            self._thresh_optim(f1_scores, thresholds, model)
        self.tprs = np.array(tprs)
        self.mean_tprs = self.tprs.mean(axis=0)
        sem = self.tprs.std(axis=0) / np.sqrt(self.ct)
        self.tprs_upper = np.minimum(self.mean_tprs + 1.96*sem, 1)
        self.tprs_lower = self.mean_tprs - 1.96*sem
        self.mean_auc = auc(self.base_fpr, self.mean_tprs)
        self.auc_upper = auc(self.base_fpr, self.tprs_upper)
        self.auc_lower = auc(self.base_fpr, self.tprs_lower)
        if self.X.shape[1] == 1:
            self.thresh_hus = np.array(self.thresh_hus)
            self.mean_thresh_hu = self.thresh_hus.mean(axis=0)
            self.sem_thresh_hu = self.thresh_hus.std(axis=0) / np.sqrt(self.ct)
        else:
            self.threshs = np.array(self.threshs)
            self.mean_thresh = self.threshs.mean(axis=0)
            self.sem_thresh = self.threshs.std(axis=0) / np.sqrt(self.ct)

    def fit_roc(self, train, test):
        np.random.shuffle(self.idx)
        model = LogisticRegression().fit(self.X[self.idx][train], self.Y[self.idx][train])
        y_score = model.predict_proba(self.X[self.idx][test])
        fpr, tpr, thresholds = roc_curve(self.Y[self.idx][test], y_score[:, 1])
        tpr = interp(self.base_fpr, fpr, tpr)
        tpr[0] = 0.0
        return model, y_score, fpr, tpr, thresholds

    def plot_roc(self, ax, color='darkorange', label='Single-energy'):
        ax.plot(self.base_fpr, self.mean_tprs, color=color,
                label=f'{label} model: AUC = {self.mean_auc:0.2f} (95% CI {self.auc_lower:0.2f} - {self.auc_upper:0.2f})')
        ax.fill_between(self.base_fpr, self.tprs_lower, self.tprs_upper, color=color, alpha=0.3);

    def sensitivity_analysis(self, test, y_score, thresholds):
        f1_scores = []
        accs = []
        for i in thresholds:
            y_pred = np.zeros(len(y_score))
            y_pred[y_score[:, 1] > i] = 1
            f1 = f1_score(self.Y[self.idx][test], y_pred)
            f1_scores.append(f1)
            acc = accuracy_score(self.Y[self.idx][test], y_pred)
            accs.append(acc)
        f1_scores = np.array(f1_scores)
        accs = np.array(accs)
        return f1_scores, accs

    def _thresh_optim(self, f1_scores, thresholds, model):
        thresh = float(np.array(thresholds[f1_scores == np.amax(f1_scores)]).mean(axis=0))
        if self.X.shape[1] == 1:
            thresh_hus = []
            thresh_hu = float((-np.log(1 / thresh - 1) - model.intercept_) / model.coef_)
            self.thresh_hus.append(thresh_hu)
        else:
            self.threshs.append(thresh)
