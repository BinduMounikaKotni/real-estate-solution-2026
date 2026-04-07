import pandas as pd
import os

def load_final_df():
    """
    Load the final engineered dataset.
    """
    path = os.path.join("final.csv")
    df = pd.read_csv(path)
    return df