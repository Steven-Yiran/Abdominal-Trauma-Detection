import os
import sys
import pandas as pd
import numpy as np
import csv
import random


ORGAN_SEG_INPUT_PATH = "../data/p_organ_images/organ_seg/df_images_train_with_seg.csv"
ORGAN_SEG_OUTPUT_PATH = "../data/p_organ_images/organ_seg/output_test.csv"
NUMBER_OF_FRAMES_PER_ORGAN = 10

def organ_based_sampleing(images):

    """
        The method takes two input parameters to generates organ-based sampling 
        images indexs

        images : all images of one patient
        num_frames_per_organ : number of images extracted for each organ for each patient
    """
    upper_bound = get_upper_bond()
    if upper_bound < (NUMBER_OF_FRAMES_PER_ORGAN * 4):
        raise ValueError("Number of images extraction over the upper bond")
    
    sampling = []
    patient_ids = get_patient_id_from_train_csv()
    for pid in patient_ids:
        liver, spleen, bowel, kidney = get_organ_sampling_images_index(patient_id = pid, 
                                                                       num_frames = NUMBER_OF_FRAMES_PER_ORGAN)
        sampling += liver + spleen + bowel + kidney
        # print("pid = ", pid)
        # print("liver = ", liver, "size = ", len(liver))
        # print("spleen = ", spleen, "size = ", len(spleen))
        # print("bowel = ", bowel, "size = ", len(bowel))
        # print("kidney = ", kidney, "size = ", len(kidney))

        # print("pid sampling = ", sampling)

    return np.array([images[i] for i in sampling])


def uniformly_random_sampling(images):
    upper_bound = get_upper_bond()
    selected_indices = set()
    while len(selected_indices) < int(upper_bound * 0.7):
        random_number = int(np.random.normal(0, 1) * upper_bound)
        random_number = np.clip(random_number, 0, upper_bound - 1)
        if random_number not in selected_indices:
            selected_indices.add(random_number)
    index_list = sorted(selected_indices)
    return np.array([images[i] for i in index_list])


def get_upper_bond():
    """
        This Method returns the Upper Bonds of images selection from `/data/pid/sid...`
    """
    table = pd.read_csv('../data/train.csv')
    bond = sys.maxsize
    number_of_pid = len(table)
    for i in range (number_of_pid):
        row = table.iloc[i]
        patient_id = row['patient_id']
        patient_path = f'../data/train_images/{patient_id}'
        series_path = os.path.join(patient_path, os.listdir(patient_path)[0])
        image_paths = sorted([os.path.join(series_path, file_name) for file_name in os.listdir(series_path)])
        bond = min(bond, len(image_paths))              
    if bond != sys.maxsize:
        return bond
    raise ValueError ("Get Upper Bonds of frames failed")

def get_patient_id_from_train_csv():
    table = pd.read_csv('../data/train.csv')
    number_of_pid = len(table)
    pid_list = []
    for i in range (number_of_pid):
        row = table.iloc[i]
        patient_id = row['patient_id']
        pid_list.append(patient_id)
    return pid_list


def get_organ_sampling_images_index(patient_id : str, num_frames : int):
    threshold = 0.95
    table = pd.read_csv(ORGAN_SEG_INPUT_PATH)
    n = len(table)
    liver = []
    spleen = [] 
    bowel = [] 
    kidney = []
    ptr = 0
    for i in range(n):
        row = table.iloc[i]
        if (patient_id == row['patient_id']):
            p_liver = float(row['pred_liver'])
            p_spleen = float(row['pred_spleen'])
            p_bowel = float(row['pred_bowel'])
            p_kidney = float(row['pred_kidney'])
            if (p_liver >= threshold) and (ptr < 50):
                liver.append(ptr)
            if (p_spleen >= threshold) and (ptr < 50):
                spleen.append(ptr)
            if (p_bowel >= threshold) and (ptr < 50):
                bowel.append(ptr)
            if (p_kidney >= threshold) and (ptr < 50):
                kidney.append(ptr)

            ## Random pick seg images 
            # if (p_liver >= threshold):
            #     liver.append(ptr)
            # if (p_spleen >= threshold):
            #     spleen.append(ptr)
            # if (p_bowel >= threshold):
            #     bowel.append(ptr)
            # if (p_kidney >= threshold):
            #     kidney.append(ptr)
            ptr += 1
        else:
            ptr = 0
    
    if len(liver) >= num_frames:
        liver = sorted(random.sample(liver, num_frames))
    if len(spleen) >= num_frames:
        spleen = sorted(random.sample(spleen, num_frames))
    if len(bowel) >= num_frames:
        bowel = sorted(random.sample(bowel, num_frames))
    if len(kidney) >= num_frames:
        kidney = sorted(random.sample(kidney, num_frames))

    # print("liver p(organ) sampling = ", liver)
    # print("spleen p(organ) sampling = ", spleen)
    # print("bowel p(organ) sampling = ", bowel)
    # print("kidney p(organ) sampling = ", kidney)
    # raise ValueError("Stop")

    return liver, spleen, bowel, kidney




    
