# Per-90 Scatter Explorer

Interactive Streamlit app to explore per-90 scatterplots for Top 5 league players. Select a player to highlight, pick any two metrics, and compare against the Top-N outliers for each axis.

## Features
- Per-90 scatterplot with highlighted player and Top-N outliers.
- Filters for season, league, position, and minimum 90s played.
- Download the current plot as a PNG.

## Project Structure
- `app.py`: Streamlit app.
- `data/working_dataset.csv`: Input dataset used by the app.

## Run Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data Refresh
Update `data/working_dataset.csv` with your latest FBref scrape, then restart the app. The UI will reflect new seasons/leagues/metrics automatically.

## Deployment
This app is designed for Streamlit Community Cloud. Use `app.py` as the entry point and make sure the dataset file is included in the repo if you want public deployment.

## Credits
Created by scoreadive.
