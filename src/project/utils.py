import pandas as pd
import os
from .config import PROJECT_DIR

def load_data():
    data_file = os.path.join(PROJECT_DIR, 'data/loan_data_2007_2014.csv')
    return pd.read_csv(data_file)


def get_actual_predicted_probs_df(y_true, y_pred_prob, thresh=0.5):

    df_actual_predicted_probs = pd.DataFrame({'true': y_true, 
                                              'pred': (y_pred_prob > thresh).astype(int), 
                                              'pred_prob': y_pred_prob}).sort_values('pred_prob').reset_index(drop=True)
    

    df_actual_predicted_probs['cumulative_n_pop'] = df_actual_predicted_probs.index + 1
    df_actual_predicted_probs['cumulative_n_good'] = df_actual_predicted_probs['true'].cumsum()
    df_actual_predicted_probs['cumulative_n_bad'] = df_actual_predicted_probs['cumulative_n_pop'] - df_actual_predicted_probs['cumulative_n_good']

    df_actual_predicted_probs['cumulative_perc_pop'] = df_actual_predicted_probs['cumulative_n_pop']/df_actual_predicted_probs.shape[0]
    df_actual_predicted_probs['cumulative_perc_good'] = df_actual_predicted_probs['cumulative_n_good']/df_actual_predicted_probs['true'].sum()
    df_actual_predicted_probs['cumulative_perc_bad'] = df_actual_predicted_probs['cumulative_n_bad']/(df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['true'].sum())


    return df_actual_predicted_probs