# NoiseAI

NoiseAI is a predictive application that uses machine learning to forecast **day and night noise levels** for different stations. The project includes a **Random Forest model** for prediction and a **web interface** to interact with the model.

---

## Features

- Train **Random Forest** models per station using historical data.
- Predict **day and night noise values** based on `year` and `month`.
- Display **available stations** and their **day/night limits** in a sidebar.
---

## Setup Instructions

```bash
git clone https://github.com/RamakotireddyGuttikonda/NoiseAI.git
cd NoiseAI

pip install -r requirements.txt

python Train_Random_Forest.py
##Already Trained Models are available in random_forest_station_models directory

python app.py

```

# You can test this application , by previewing live server on index.html

---
## Dataset Used 

https://www.kaggle.com/datasets/rohanrao/noise-monitoring-data-in-india

---

You can Contact me through ramakotireddyguttikonda8@gmail.com
-Leave a star if u like the project

