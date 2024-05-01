



def generate(organ_classification_model, injury_classification_model, sample):
    dataset = RawDataset()

    for i in range(len(dataset)):
        (
            images,
            bowel_healthy,
            extravasation_healthy,
            kidney_condition,
            liver_condition,
            spleen_condition
        ) = dataset[i]

        sampled_images = sample(images)

        p_organs = organ_classification_model(sampled_images)
        p_injuries = injury_classification_model(sampled_images)

    # save (p_organs, p_injuries, label) to disk

if __name__ == '__main__':
    organ_classification_model = ... # load from saved model
    injury_classification_model = ...

    generate(organ_classification_model, injury_classification_model)
