import numpy as np
import pandas as pd
from sklearn.metrics import auc
from scipy.stats import norm
from statsmodels.stats.anova import AnovaRM
from performance import exact_binomial_conf # import from local performance.py file


def binary_kappa_table(a, b):
    n = a.shape[0]
    assert n == b.shape[0]
    k_tab = np.zeros((2, 2))
    k_tab[0, 0] = ((a == 'ca') & (b == 'ca')).sum()
    k_tab[0, 1] = ((a == 'ca') & (b == 'hem')).sum()
    k_tab[1, 0] = ((a == 'hem') & (b == 'ca')).sum()
    k_tab[1, 1] = ((a == 'hem') & (b == 'hem')).sum()
    return k_tab

def reader_accuracy(df:pd.core.frame.DataFrame, task_r='mix_p'):
    accs = []
    for i in df.index:
        if (df.loc[i, task_r] == 'ca') & (df.loc[i, 'ca'] == 1):
            accs.append(1)
        elif (df.loc[i, task_r] == 'hem') & (df.loc[i, 'hem'] == 1):
            accs.append(1)
        else:
            accs.append(0)
    return np.array(accs)

def roc_analysis(df:pd.core.frame.DataFrame, assess, reader):
    tpr = []
    fpr = []
    s = "likert_%s_%s" % (assess, reader)
    for thresh in range(-2, 3, 1):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in df.index:
            if df.loc[i, 'ca'] == 1:
                if df.loc[i, s] < thresh:
                    tn += 1
                else:
                    fp += 1
            else:
                if df.loc[i, s] < thresh:
                    fn += 1
                else:
                    tp += 1
        tpr.append(tp / (tp + fn))
        fpr.append(1 - (tn / (tn + fp)))
    tpr.append(0)
    fpr.append(0)
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    rauc = auc(fpr, tpr)
    return tpr, fpr, rauc


class ReaderConfidence():
    def __init__(self, data:pd.core.frame.DataFrame, reader='p'):
        self.data = data
        self.reader = reader
        self.n = len(data)
        self.se_conf = self._confidence_table(series='Single-energy')
        self.de_conf = self._confidence_table(series='Dual-energy')
        self.reader = 2 if reader == 's' else 1

    def analyze(self):
        self.calculate_stats()
        self.print_stats()
        self.conf = pd.concat([self.se_conf, self.de_conf])
        model = AnovaRM(self.conf, depvar='conf', subject='lesion', within=['series'])
        self.result = model.fit()
        print(f'Repeated Measures ANOVA results for Reader #{self.reader}:')
        print(self.result.summary())

    def calculate_stats(self):
        self.se_n_cert = self.se_conf.conf.sum()
        self.se_pct_cert = self.se_n_cert / self.n
        se_rng = exact_binomial_conf(self.se_pct_cert, self.n)
        se_lower = (self.se_pct_cert - se_rng) * self.n
        se_upper = (self.se_pct_cert + se_rng) * self.n
        self.se_ci = f'{se_lower:.0f} - {se_upper:.0f}'
        self.de_n_cert = self.de_conf.conf.sum()
        self.de_pct_cert = self.de_n_cert / self.n
        de_rng = exact_binomial_conf(self.de_pct_cert, self.n)
        de_lower = (self.de_pct_cert - de_rng) * self.n
        de_upper = (self.de_pct_cert + de_rng) * self.n
        self.de_ci = f'{de_lower:.0f} - {de_upper:.0f}'

    def print_stats(self):
        se_str = f'{self.se_n_cert}/{self.n} ({self.se_pct_cert * 100:.0f}%, 95% CI: {self.se_ci})'
        de_str = f'{self.de_n_cert}/{self.n} ({self.de_pct_cert * 100:.0f}%, 95% CI: {self.de_ci})'
        print(f'Reader #{self.reader} "certain" on Single-energy: {se_str}')
        print(f'Reader #{self.reader} "certain" on Dual-energy: {de_str}')
        print()

    def _confidence_table(self, series='Single-energy'):
        if series == 'Dual-energy':
            col_str = f'conf3_{self.reader}'
        else:
            col_str = f'conf1_{self.reader}'
        conf = {idx: 1 if conf == 2 else 0 for idx, conf in self.data[col_str].iteritems()}
        conf = pd.Series(conf)
        conf = pd.DataFrame({'conf': conf}).reset_index().rename(columns={'index': 'lesion'})
        conf['series'] = series
        return conf
