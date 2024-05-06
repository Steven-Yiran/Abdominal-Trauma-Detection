import os
import sys
import pandas as pd
import numpy as np
import csv
import random


ORGAN_SEG_INPUT_PATH = "../data/p_organ_images/organ_seg/df_images_train_with_seg.csv"
NUMBER_OF_LABELS = 4
THRESHOLD = 0.95
MAX_NUM_FRAMES = 80

def collectSamplingData(metadata, strategy):
    match strategy:
        case 'Uniform':
            return uniformSampling(metadata)
        case 'OrganBased':
            return organBasedSampling(metadata)
        case _:
            raise ValueError ('Invalid Strategy Name')
            
def organBasedSampling(metadata):
    
    table = pd.read_csv(ORGAN_SEG_INPUT_PATH)
    request_patient_id = metadata['patient_id']
    request_patient_series_id = metadata['series_id']

    liver = []
    spleen = [] 
    bowel = [] 
    kidney = []
    pointer = 0
    print("Applying sampling to patient ID:",request_patient_id, ",with series ID:",request_patient_series_id)
    for g_index in range(len(table)):
        row = table.iloc[g_index]
        if ((row['patient_id'] == request_patient_id) and (row['series'] == request_patient_series_id)):
            if float(row['pred_liver']) >= THRESHOLD:
                liver.append(pointer)
            if float(row['pred_spleen']) >= THRESHOLD:
                spleen.append(pointer)
            if float(row['pred_bowel']) >= THRESHOLD:
                bowel.append(pointer)
            if float(row['pred_kidney']) >= THRESHOLD:
                kidney.append(pointer)
            pointer += 1
        else:
            pointer = 0
    n_frame = int(MAX_NUM_FRAMES/4)
    liver = random.sample(liver, min(n_frame,len(liver)))
    spleen = random.sample(spleen, min(n_frame,len(spleen)))
    bowel = random.sample(bowel, min(n_frame,len(bowel)))
    kidney = random.sample(kidney, min(n_frame, len(kidney)))
    print("Organ Based Sampling Completed.")
    return sorted(liver + spleen + bowel + kidney)


def uniformSampling(metadata):
    print("Applying sampling to patient ID:",metadata['patient_id'], ",with series ID:",metadata['series_id'])
    frames = getNumberOfSegImages(metadata)
    frames = min(frames, MAX_NUM_FRAMES)
    selected_indices = set()
    while len(selected_indices) < int(frames * 0.7):
        random_number = int(np.random.normal(0, 1) * frames)
        random_number = np.clip(random_number, 0, frames - 1)
        if random_number not in selected_indices:
            selected_indices.add(random_number)
    index_list = sorted(selected_indices)
    print("Uniform Sampling Completed.")
    return index_list

def getNumberOfSegImages(data):
    table = pd.read_csv(ORGAN_SEG_INPUT_PATH)
    request_patient_id = data['patient_id']
    request_patient_series_id = data['series_id']
    counter = 0
    for i in range(len(table)):
        row = table.iloc[i]
        if ((row['patient_id'] == request_patient_id) and (row['series'] == request_patient_series_id)):
            counter += 1
    return counter














    
