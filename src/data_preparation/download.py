import os

from kaggle import api


def download_from_competition(filename, path):
    api.competition_download_file(
        competition='rsna-2023-abdominal-trauma-detection',
        file_name=filename,
        path=path
    )

def download_theo(
    dataset,
    patient_id,
    series_id,
    number_of_images
):
    path = f'../data/train_images/{patient_id}/{series_id}'
    os.makedirs(path)

    failed_image_ids = []

    for i in range(number_of_images):
        image_id = f'{i:04d}'
        file_name = f'{patient_id}_{series_id}_{image_id}.png'
        
        try:
            api.dataset_download_file(
                dataset=dataset,
                file_name=file_name,
                path=path
            )

            current_name = os.path.join(path, file_name)
            new_name = os.path.join(path, f'{image_id}.png')
            os.rename(current_name, new_name)
        except:
            failed_image_ids.append(image_id)
    
    print('Failed to download these images:')
    print(failed_image_ids)

if __name__ == '__main__':
    
    download_theo(
        dataset='theoviel/rsna-abdominal-trauma-detection-png-pt2',
        patient_id='10005',
        series_id='18667',
        number_of_images=103
    )

    download_theo(
        dataset='theoviel/rsna-abdominal-trauma-detection-png-pt4',
        patient_id='10007',
        series_id='47578',
        number_of_images=116
    )