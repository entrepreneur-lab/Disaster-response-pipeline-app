import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)
# important to allow use of zip in jinja template
app.jinja_env.filters['zip'] = zip

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCategories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = [i.title() for i in list(genre_counts.index)]

    # extract data for 2nd visual
    class_distr1 = df.drop(['id', 'message', 'original', 'genre'],
                           axis=1).sum()/len(df)

    # sort values in ascending
    class_distr1 = class_distr1.sort_values(ascending=False)

    # series of values that have 0 in classes
    class_distr0 = (class_distr1 -1) * -1
    class_name = [i.replace('_', ' ').title() for i in list(class_distr1.index)]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        # data and layout for 2nd visual
        {
            'data': [
                Bar(
                    x=class_name,
                    y=class_distr1,
                    name = 'Class = 1',
                    marker = dict(
                        color='rgb(252, 34, 34)'
                        ),
                ),
                Bar(
                    x=class_name,
                    y=class_distr0,
                    name='Class = 0',
                    marker=dict(
                            color='rgb(246, 131, 131)'
                                ),
                )
            ],

            'layout': {
                'title': 'Distribution of labels within classes',
                'yaxis': {
                    'title': "Distribution",
                },
                'xaxis': {
                    'title': "",
                    'tickangle': -45,
                },
                'barmode' : 'stack',
                'margin' : {
                    'b' : 100
                    }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
