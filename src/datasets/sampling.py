import os
import sys
import pandas as pd
import numpy as np
import csv
import random


def collectSamplingData(sampling_config):
    match sampling_config.strategy:
        case 'Uniform':
            return uniformSampling(sampling_config)
        case 'OrganBased':
            return organBasedSampling(sampling_config)
        case _:
            raise ValueError ('Invalid Strategy Name')
            
def organBasedSampling(config):
    table = pd.read_csv(config.organSegPath)
    request_patient_id = config.metadata['patient_id']
    request_patient_series_id = config.metadata['series_id']

    liver = []
    spleen = [] 
    bowel = [] 
    kidney = []
    pointer = 0
    print("Applying sampling to patient ID:",request_patient_id, ",with series ID:",request_patient_series_id)
    for g_index in range(len(table)):
        row = table.iloc[g_index]
        if ((row['patient_id'] == request_patient_id) and (row['series'] == request_patient_series_id)):
            if float(row['pred_liver']) >= config.threshold:
                liver.append(pointer)
            if float(row['pred_spleen']) >= config.threshold:
                spleen.append(pointer)
            if float(row['pred_bowel']) >= config.threshold:
                bowel.append(pointer)
            if float(row['pred_kidney']) >= config.threshold:
                kidney.append(pointer)
            pointer += 1
        else:
            pointer = 0
    n_frame = int(config.numFrames/4)
    liver = random.sample(liver, min(n_frame,len(liver)))
    spleen = random.sample(spleen, min(n_frame,len(spleen)))
    bowel = random.sample(bowel, min(n_frame,len(bowel)))
    kidney = random.sample(kidney, min(n_frame, len(kidney)))
    print("Organ Based Sampling Completed.")
    print("organ list = ", liver + spleen + bowel + kidney)
    return sorted(liver + spleen + bowel + kidney)


def uniformSampling(config):
    print("Applying sampling to patient ID:",config.metadata['patient_id'], ",with series ID:",config.metadata['series_id'])
    frames = getNumberOfSegImages(config)
    frames = min(frames, config.numFrames)
    selected_indices = set()
    while len(selected_indices) < int(frames * 0.7):
        random_number = int(np.random.normal(0, 1) * frames)
        random_number = np.clip(random_number, 0, frames - 1)
        if random_number not in selected_indices:
            selected_indices.add(random_number)
    index_list = sorted(selected_indices)
    print("Uniform Sampling Completed.")
    return index_list

def getNumberOfSegImages(config):
    table = pd.read_csv(config.organSegPath)
    request_patient_id = config.metadata['patient_id']
    request_patient_series_id = config.metadata['series_id']
    counter = 0
    for i in range(len(table)):
        row = table.iloc[i]
        if ((row['patient_id'] == request_patient_id) and (row['series'] == request_patient_series_id)):
            counter += 1
    return counter














    
