# Basketball Performance Predictor App

This repository contains the code for a Streamlit web app that predicts a basketball player's Player Efficiency Rating (PER) based on their statistics.

The model behind the app was trained using a dataset of historical player statistics from 1990 onwards, which includes attributes like shooting efficiency, three-point efficiency, free throw efficiency, rebound rate, assist rate, and turnover rate.

## Dependencies

All the dependencies for this project are listed in the `requirements.txt` file. They can be installed with the following command:

```bash
pip install -r requirements.txt
```

## Usage
To run the app locally, navigate to the directory containing the app files and type the following command into your terminal:

```bash
streamlit run per_predictor_app.py
```

The app is also hosted on Streamlit. You can visit the app at [insert your Streamlit app's URL here].

## Project Files
*Seasons_Stats.csv: This is the dataset used to train the model. It contains historical basketball player statistics from 1990 onwards.

per_predictor_app.py: This is the Streamlit app script. It handles user input and output, and loads and uses the pre-trained PyTorch model to make predictions.

final_model.pt: This is the pre-trained PyTorch model. It was trained using the PyTorch framework on the dataset contained in Seasons_Stats.csv.

scaler.pkl: This is the saved instance of StandardScaler that was used to standardize the training data before it was fed to the model. It is loaded by per_predictor_app.py to standardize the user input before making a prediction.

## Author
Eric Grose

License
This project is licensed under the MIT License.


Please replace `[Your Name]` and `[insert your Streamlit app's URL here]` 
