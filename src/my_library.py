
# helper function that set up inital imports
def make_inital_imports():
    import pandas as pd
    import numpy as np
    #import scipy.stats as stats
    import matplotlib.pyplot as plt
    #import matplotlib.ticker as ticker
    import seaborn as sns
    
    return pd, np, plt, sns

# helper function to load a dataset
def load_data(url):
    # Load CSV file with ; seperator
    df = pd.read_csv(url, sep=';')
    return df