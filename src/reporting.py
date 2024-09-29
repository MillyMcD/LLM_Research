import pandas as pd
import numpy as np

from pathlib import Path

def folder_to_dataframe(pth:Path,model_list:list):
    """
    Unpack a Verifier output folder into a pandas dataframe.

    Parameters
    ----------
    pth : `str`
        path to folder
    model_list : list[str]
        List of models to include

    Returns
    -------
    `pd.DataFrame`
        dataframe with results
    """

    #loop through folders in verifier folder, then loop through files to grab results
    #then unpack results into a dictionary
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
                          'tps':df['mean_tps'].mean(),
                          'sem_accuracy':df['sem_acc'].mean()}
            
            records[file.stem][f'{dir.name}_accuracy'] = df['accuracy'].mean()

    #tidy up
    nrecords = []
    acc_columns = None
    for k,v in records.items():
        rec = {k1:v1 for k1,v1 in v.items()}
        avg = [v1 for k1,v1 in v.items() if 'accuracy' in k1]
        if acc_columns is None:
            acc_columns = sorted([k1 for k1,v1 in rec.items() if 'accuracy' in k1])
        rec['average_accuracy'] = np.mean(avg).item()
        nrecords.append(rec)
    acc_columns.append('average_accuracy')
    df = pd.DataFrame(nrecords)
    df = df[['model','time','tps',*acc_columns]]
    return df


def filter_folder_to_dataframe(pth:Path,model_list:list,ids:list):
    """
    Unpack a Verifier output folder into a pandas dataframe, but now 
    filtering the questions by ID.

    Parameters
    ----------
    pth : `str`
        path to folder
    model_list : list[str]
        List of models to include

    Returns
    -------
    `pd.DataFrame`
        dataframe with results
    """
    #loop through folders in verifier folder, then loop through files to grab results
    #then unpack results into a dictionary
    records = {}
    for dir in Path(pth).iterdir():
        if not dir.name in ['gemma2','llama3.1']:
            continue
        print(dir)
        for file in dir.iterdir():
            if file.stem not in model_list:
                continue
            df = pd.read_csv(file)
            df = df.loc[df.id.isin(ids)]

            record = {f'{dir.name}_accuracy': df['consistent'].sum()/len(df)}
            record['sem_accuracy'] = (df['sem_score']>0.7).sum()/len(df)
            record['model'] = file.stem
            #record[f'{dir.name}_df'] = df
            if file.stem in records.keys():
                records[file.stem].update(record)
            else:
                records[file.stem] = record
    #tidy up
    nrecords = []
    acc_columns = None
    for k,v in records.items():
        rec = {k1:v1 for k1,v1 in v.items()}
        avg = [v1 for k1,v1 in v.items() if 'accuracy' in k1]
        if acc_columns is None:
            acc_columns = sorted([k1 for k1,v1 in rec.items() if 'accuracy' in k1])
        rec['average_accuracy'] = np.mean(avg).item()
        nrecords.append(rec)
    acc_columns.append('average_accuracy')
    df = pd.DataFrame(nrecords)
    return df