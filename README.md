# Disaster Response Pipeline Project

### Introduction
This app was developed to classify text into disaster response categories and identify services which might be needed based on the results. The dataset was provided by Figure Eight and supplied through Udacity as part of the Data Science Nanodegree. A brief overview of the data is shown on the homepage. These visualisations show a class imbalance in the dataset, which I have attempted to correct in the model design.

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Once you have created the database and trained your model, run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project Structure
```- app
| - template
| |- master.html            # main page of web app
| |- go.html                # classification result page of web app
|- run.py                   # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv    # data to process
|- process_data.py          # script to process the text data and create the database
|- InsertDatabaseName.db    # database of processed data (generated after running process_data.py)

- models
|- train_classifier.py      # script to train the model
|- classifier.pkl           # saved model (generated after running train_classifier.py)

- README.md```
