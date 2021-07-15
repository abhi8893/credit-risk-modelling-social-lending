import pandas as pd
import os
from .config import PROJECT_DIR

def load_data():
    data_file = os.path.join(PROJECT_DIR, 'data/loan_data_2007_2014.csv')
    return pd.read_csv(data_file)
