import os, sys
import pandas as pd
import numpy as np
import pickle

def get_sub_dir (sub_dir_name, home_dir_name = 'RL'):
    cwdir = os.getcwd()
    pos = cwdir.find(home_dir_name)
    assert pos > -1, f"No directory with the name {home_dir_name} exists!"
    
    home_dir = cwdir[:pos] + home_dir_name
    sub_dir = os.path.join(home_dir, sub_dir_name)
    assert os.path.isdir(sub_dir), f"{sub_dir} is NOT a valid directory!"

    return sub_dir

def save_pickle (fp, obj):
    with open(fp, 'wb') as file:
        pickle.dump(obj, file)
        print(f"Object saved to {fp}")
        
def load_pickle (fp):
    assert os.path.isfile(fp), "Not a valid filepath!"
    with open(fp, 'rb') as file:
        return pickle.load(file)    

def sneak_peek (df, size=3, check_null=False):
    print(df.shape)
    if check_null:
        print(df.isnull().sum())
    return pd.concat([df.head(size), df.tail(size)])

def overview (df, size=3, check_null=False):
    print(df.shape)
    if check_null:
        num_nulls = df.isnull().sum().sum()
        print(f"Number of nulls: {num_nulls}")
    return pd.concat([df.head(size), df.tail(size)])

def get_ifg_dir (version):
    data_dir = get_sub_dir('data')
    return f"{data_dir}/ifg_v{version}"

def get_ifg_file (version, filename):
    data_dir = get_sub_dir('data')
    fdir = f"{data_dir}/ifg_v{version}"
    filepath = os.path.join(fdir, filename)
    assert os.path.isfile(filepath), "Invalid file name!"

    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    elif filepath.endswith(('.pkl', 'pickle')):
        return pd.read_pickle(filepath)
    else:
        print(f"File type {filepath} not found!")
    
# def main():
#     # Your code here
#     pass


# if __name__ == "__main__":
#     main()