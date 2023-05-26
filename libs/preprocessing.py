import os
import re
import pandas as pd
from functools import reduce
import numpy as np
import argparse
import json

def remove_outliers(df, fri_hh=1e4, ss_cutoff=200, pt_cutoff=500):
    print("===== 2. remove_outliers =====")

    for c in df.columns:
        x = df[c].values
        if "FRI" in c:
            df[c] = x * ((x>=0) & (x < fri_hh))
        elif "TEI" in c:
            df[c] = x * ((x>=10) & (x <50))

    return df

def merge_data_frames(dfs): 
    return reduce(lambda df, ndf: pd.merge(df, ndf, left_index=True, right_index=True, how="left"), dfs[1:], dfs[0])

def read_csv_files_from_regex(target_dir, expr, **options):
    files = os.listdir(target_dir)
    m = re.compile(expr)
    result = []

    for file in files:
        if not m.match(file): continue

        _file = os.path.join(target_dir, file)
        df = pd.read_csv(_file, index_col=0)
        col_name = options["col_identify_func"](file)
        df.columns = [col_name]

        if "filter_func" in options: df = options["filter_func"](df)
        if df.index.size == 0: continue
        
        result.append( df )
        print(len(df.index))

    return result

def check_missing_rows(df, interval_minutes):
    return;    

def preprocessing(df):
    df.index = pd.to_datetime(df.index)
    df = remove_outliers(df)
    df = df.resample('1min').last().fillna(0)
    df = time2vec(df)
    return df

def time2vec(df):
    df2 = df.copy()
    d = df2.index.weekday
    h = df2.index.hour
    m = df2.index.minute
    
    week_minute = ((d*1440) + (h*60) + m)
    df2.loc[:,'T2WS'] = np.sin(week_minute * 2*np.pi / (1440*7))
    df2.loc[:,'T2WC'] = np.cos(week_minute * 2*np.pi / (1440*7))
    df2.loc[:,'T2DS'] = np.sin(week_minute * 2*np.pi / 1440)
    df2.loc[:,'T2DC'] = np.cos(week_minute * 2*np.pi / 1440)
    return df2

def load_config(config_dir, session):
    with open(config_dir) as f:
        config = json.load(f)
        session = config[session] 

    return session  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--session')
    parser.add_argument('--start_date')    
    parser.add_argument('--end_date')    

    args = parser.parse_args()

    config = load_config("defpipe.json", args.session)

    args.start_date += " 00:00:00"
    args.end_date += " 23:59:59"

    start_date, end_date = args.start_date, args.end_date

    include_regex = config["include_regex"]
    base_dir = config["base_dir"]
    output_path = config["output_path"]

    options = {
        "col_identify_func": lambda x: x.replace("KSSCADA.701-367-", "").replace(".F_CV.csv", ""),
        "filter_func": lambda df: df.loc[start_date:end_date],
    }
    
    options = {
        "col_identify_func": lambda x: x.replace("DJEPGS0.701-", "").replace(".F_CV.csv", ""),
        "filter_func": lambda df: df.loc[start_date:end_date],
    }
    
    dfs = read_csv_files_from_regex(base_dir, include_regex, **options)
    df = merge_data_frames(dfs)

    for c in config["drop_columns"]: df = df.drop(c, axis=1)

    df = preprocessing(df)
    print(df.index.size)

    df.to_csv(output_path)

