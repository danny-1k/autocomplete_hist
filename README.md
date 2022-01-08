# Autocomplete

Extremely simple(and maybe useless) LSTM model trained on google search history data

## Getting started
1. <a href='https://takeout.google.com'>Get your google search data (only google search)</a>
2. Put the MyActivity.html file in the `data` folder
3. Generate the JSON `python src/save_to_json.py`
4. `cd src`
5. Create dataset `python process_data.py --past_count 4 --freq_tresh 5 --window 1`
6. Train cbow model (for embeddings) `python train_cbow.py`
7. Train autocomplete model `python train.py`
8. Run web app `python app.py`


## Demo

https://user-images.githubusercontent.com/88403488/148645106-86a53218-bc4b-4b1d-9a24-0b4ec3c2b0f4.mp4
