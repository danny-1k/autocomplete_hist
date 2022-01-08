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

[noodle_demo_one_word.mp4](https://github.com/danny-1k/autocomplete_hist/blob/2105318e92a9930c7b2011f8dc2d421df9f85f7a/noodle_demo_one_word.mp4)