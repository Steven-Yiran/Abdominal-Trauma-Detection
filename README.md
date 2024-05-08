# Abdominal Trauma Detection

## How to Run

### Data Preparation

Download the [dataset](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection) from Kaggle to the `data` directory. Place `train.csv` in `data`. Place the CT scan images in PNG format in `data/train_images`, and organize them based on patient ID and series ID (`data/train_images/{patient_id}/{series_id}/{image_id_in_series}.png`).

### Model Training

The training config is in `./src/main.py`.

To train the classification model, run

```
python main.py --train-classifier
```

To prepare the data for training the LSTM, run

```
python main.py --frame-inference
```

To train the LSTM, run

```
python main.py --train-rnn
```

To run the above end-to-end, run

```
python main.py --end-to-end
```

## Our Results

We trained on CT scans of 400 patients (about 14GB in total) for 10 epochs with one GPU on Oscar. The script and outputs are in `./oscar_training`.