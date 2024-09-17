import torch
import numpy as np
from sklift.metrics import qini_auc_score
import pandas as pd

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 使用确定性卷积算法
    torch.backends.cudnn.benchmark = False  # 关闭cudnn自适应算法，确保可复现性

def cal_ltv_qini(ltv_trues, ltv_preds):
    data = {
        'ltv_trues': ltv_trues,
        'ltv_preds': ltv_preds
    }
    df = pd.DataFrame(data)
    sums = df['ltv_trues'].sum()
    df['ltv_trues'] = df['ltv_trues'] / sums
    sorted_df_pred = df.sort_values(by='ltv_preds', ascending=False)
    auc_pred = sorted_df_pred['ltv_trues'].cumsum().sum()
    sorted_df_true = df.sort_values(by='ltv_trues', ascending=False)
    auc_true = sorted_df_true['ltv_trues'].cumsum().sum()
    qini = auc_pred / auc_true

    df = pd.DataFrame(data)
    bins, _ = pd.qcut(df['ltv_preds'], q=10, labels=False, retbins=True, duplicates='drop')
    df['bkt'] = bins
    bucket_means = df.groupby('bkt')[['ltv_preds', 'ltv_trues']].mean()
    mape10 = abs(bucket_means['ltv_preds'] - bucket_means['ltv_trues']) / bucket_means['ltv_trues']

    return qini, mape10.mean(), bucket_means

def cal_auuc(y_true, uplift, treatment):
    qini = 0
    cnt = 0
    for i in y_true.keys():
        cnt += len(y_true[i])
        qini += qini_auc_score(y_true[i], uplift[i], treatment[i] * len(y_true[i]))
    return qini / cnt