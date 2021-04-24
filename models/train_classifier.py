# import libraries
import re
import sys
import pickle
import pandas as pd
import nltk
nltk.download(['punkt','wordnet'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support

def load_data(database_filepath):
    """
    Load the tweet data from the database

    Parameters
    ----------
    database_filepath : string
        path to the database file containing tweet data

    Returns
    -------
    X : Pandas dataframe
        contains unprocessed tweet data
    Y : Pandas dataframe
        category data for each tweet
    category_names : list of strings
        list of the possible categories for each tweet

    """
    engine = create_engine('sqlite:///' + str (database_filepath))
    df = pd.read_sql ('SELECT * FROM MessageCategories', engine)
    X = df ['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Process raw text into tokenized data for training (feature extraction)

    Parameters
    ----------
    text : string
        tweet in string format

    Returns
    -------
    cleaned_tokens : list of tokenized strings

    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for token in tokens:
        clean = lemmatizer.lemmatize(token).lower().strip()
        cleaned_tokens.append(clean)
    return cleaned_tokens


def build_model():
    """
    Build a Random Forest Classifier on tweet data from the database

    Returns
    -------
    model : sklearn classifier
        Random forest classifier pipeline which tokenizes and extracts
        features for making predictions

    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance on the test data

    Parameters
    ----------
    model : sklearn classifier
        Random forest classifier pipeline which tokenizes and extracts
        features for making predictions
    X_test : numpy array
        feature data for each sample in the dataset
    Y_test : numpy array
        classification info for each sample in the dataset
    category_names : list of strings
        list of the possible categories for each tweet

    Returns
    -------
    None.

    """
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test[col],
                                                                    y_pred[:, i],
                                                                    average='weighted')

        print(f'\nReport for the column ({col}):\n')        
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F-score: {fscore:.2f}')


def save_model(model, model_filepath):
    """
    Save the model for future use

    Parameters
    ----------
    model : sklearn classifier
        Random forest classifier pipeline which tokenizes and extracts
        features for making predictions
    model_filepath : string
        directory path for saving the model

    Returns
    -------
    None.

    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
