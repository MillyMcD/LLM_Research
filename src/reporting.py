import pandas as pd
import numpy as np

from pathlib import Path

def folder_to_dataframe(pth:Path,model_list:list):
    records = {}
    for dir in Path(pth).iterdir():
        if not dir.name in ['gemma2','llama3.1']:
            continue
        for file in dir.iterdir():
            if file.stem not in model_list:
                continue
    
            df = pd.read_csv(file)
            if file.stem not in records.keys():
                records[file.stem] = {f'{dir.name}_accuracy':df['accuracy'].mean(),
                          'model':file.stem,
                          'time':df['mean_time'].mean(),
                          'tps':df['mean_tps'].mean()}
            
            records[file.stem][f'{dir.name}_accuracy'] = df['accuracy'].mean()

    nrecords = []
    acc_columns = None
    for k,v in records.items():
        rec = {k1:v1 for k1,v1 in v.items()}
        avg = [v1 for k1,v1 in v.items() if 'accuracy' in k1]
        if acc_columns is None:
            acc_columns = [k1 for k1,v1 in rec.items() if 'accuracy' in k1]
        rec['average_accuracy'] = np.mean(avg).item()
        nrecords.append(rec)
    acc_columns.append('average_accuracy')
    df = pd.DataFrame(nrecords)
    df = df[['model','time','tps',*acc_columns]]
    return df