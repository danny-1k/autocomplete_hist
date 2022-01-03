# Autocomplete

Extremely simple(and maybe useless) LSTM model trained on google search history data

## Getting started
1. <a href='https://takeout.google.com'>Get your google search data (only google search)</a>
2. Put the MyActivity.html file in the `data` folder
3. Generate the JSON `python src/save_to_json.py`
4. Create dataset `python src/process_data.py --past_count 4 --freq_tresh 5 --window 1`
5. Train cbow model (for embeddings) `python src/train_cbow.py`
6. Train autocomplete model `python src/train.py`
7. Run web app `python src/app.py`
