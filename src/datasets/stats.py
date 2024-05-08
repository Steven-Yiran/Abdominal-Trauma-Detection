import pandas as pd

def get_organ_counts(config):
    """Get the number of frames per organ.
    """
    df = pd.read_csv(config.segmentations_csv)
    df = df[['patient_id', 'series', 'frame', 'pred_liver', 'pred_spleen', 'pred_kidney', 'pred_bowel']]
    # count number of organs by thresholding the predictions on 0.5
    df['pred_liver'] = df['pred_liver'] > 0.5
    df['pred_spleen'] = df['pred_spleen'] > 0.5
    df['pred_kidney'] = df['pred_kidney'] > 0.5
    df['pred_bowel'] = df['pred_bowel'] > 0.5

    liver_count = df['pred_liver'].sum()
    spleen_count = df['pred_spleen'].sum()
    kidney_count = df['pred_kidney'].sum()
    bowel_count = df['pred_bowel'].sum()

    print(len(df))
    print(f'Liver: {liver_count}, Spleen: {spleen_count}, Kidney: {kidney_count}, Bowel: {bowel_count}')