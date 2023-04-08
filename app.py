from flask import Flask, request, app,render_template, redirect, url_for
from flask import Response
import pickle
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
import graphviz
import pydotplus
from sklearn import tree
import joblib




application = Flask(__name__)
app=application

rf_model = joblib.load('Model/random_forest_best_fixed.sav')

df = pd.read_csv("Dataset/depression_classification_final_clean_dataset.csv", index_col=0)
rf_features = ['speech_speed', 'avg_characters', 'avg_nouns', 'how are you at controlling your temper', 'when was the last time you argued with someone and what was it about']
rf_data = df[rf_features]
explainer = lime.lime_tabular.LimeTabularExplainer(rf_data.values,feature_names=rf_features,class_names=['not-depressed', 'depressed'],verbose=True,mode='classification')#discretize_continuous=True?

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

#route for final assessment page
@app.route('/finalassessment')
def finalassessment():
    return render_template('finalassessment.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    #clear the static/visualizatons folder
    folder = 'static/visualizations'
    for filename in os.listdir(folder):
        os.remove(os.path.join(folder, filename))
    return render_template('home.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    #clear the static/visualizatons folder 
    folder = 'static/visualizations'
    for filename in os.listdir(folder):
        os.remove(os.path.join(folder, filename))

    result=""

    if request.method=='POST':

        Speech_Speed=float(request.form.get("Speech_Speed"))
        avg_characters = float(request.form.get('avg_characters'))
        avg_nouns = float(request.form.get('avg_nouns'))
        temper_control_sentiment = float(request.form.get('temper_control_sentiment'))
        recency_of_argument_sentiment = float(request.form.get('recency_of_argument_sentiment'))

        new_data=np.array([[Speech_Speed,avg_characters,avg_nouns,temper_control_sentiment,recency_of_argument_sentiment]])
        prediction=rf_model.predict(new_data)
        probability=rf_model.predict_proba(new_data)    #returns the probability of the prediction as an array  [[0.9,0.1]] 0.9 is the probability of 0 and 0.1 is the probability of 1
        #reshape the array to work with the explainer
        new_data=new_data.reshape(-1)

        exp = explainer.explain_instance(new_data, rf_model.predict_proba, num_features=5)
        #get the baseline probability, local prediction and feature contributions.
        baseline_prob = exp.intercept[1]
        local_prediction = exp.local_pred
        feature_contrib = exp.as_list()

        # create the note content
        note_content = "<h3>Explanation</h3>"
        note_content += "<ul>"
        note_content += "<li>The baseline probability of a person being depressed regardless of their features is " + str(baseline_prob) + ".</li>"
        note_content += "<li>The predicted probability of depression for the specific input instance being explained is represented by the prediction local value of " + str(local_prediction) + ".</li>"
        note_content += "<li>This list below represents the contribution of each feature to the final prediction. A positive value means that the feature increases the predicted probability of depression, while a negative value means that the feature decreases the predicted probability of depression. The list is given as:</li>"
        note_content += "<ol>"
        for contrib in feature_contrib:
            note_content += "<li>"
            note_content += "<ul>"
            note_content += "<li>Feature: " + str(contrib[0]) + "</li>"
            note_content += "<li>Contribution: " + str(contrib[1]) + "</li>"
            note_content += "</ul>"
            note_content += "</li>"
        note_content += "</ol>"
        note_content += "</ul>"


        # wrap the note content in HTML tags
        limenotes = "<html><body>" + note_content + "</body></html>"

        #check if the file exists, if it does delete it
        if os.path.exists('static/visualizations/lime_explanation.html'):
            os.remove('static/visualizations/lime_explanation.html')
            #remove limenotes.html as well
            os.remove('static/visualizations/limenotes.html')
            exp.save_to_file('static/visualizations/lime_explanation.html')
            # save the note content to a file
            with open('static/visualizations/limenotes.html', 'w') as f:
                f.write(limenotes)
        #save the explanation plot
        else:
            exp.save_to_file('static/visualizations/lime_explanation.html')
            # save the note content to a file
            with open('static/visualizations/limenotes.html', 'w') as f:
                f.write(limenotes)

        #check if decision_tree.svg exists, if it does do not create it again
        if os.path.exists('static/visualizations/decision_tree.svg'):
            pass
        else:
            #create a decision tree
            tree_model = rf_model.estimators_[99]
            #plot the decision tree
            dot_data = tree.export_graphviz(tree_model, out_file=None,
                                            feature_names=rf_features,
                                            class_names=['not-depressed', 'depressed'],
                                            filled=True, rounded=True,
                                            special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render("static/visualizations/decision_tree", format="svg")
        
        #check if rf_feature_importances.svg exists, if it does do not create it again
        if os.path.exists('static/visualizations/rf_feature_importances.svg'):
            pass
        else:
            plt.barh(rf_features, rf_model.feature_importances_)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title('Random Forest Model Feature Importances')
            plt.savefig('static/visualizations/rf_feature_importances.svg', bbox_inches='tight')






        if prediction[0] ==1 :
            result = 'Depressed'
            probability=probability[0][1]
        else:
            result ='Not-Depressed'
            probability=probability[0][0]
            
        return render_template('home.html',result=result, probability=str(round((probability*100),2)), baseline_prob=baseline_prob,local_prediction=local_prediction, feature_contrib=feature_contrib ) # the result and probability can be accessed in javascript using result and probablity.

    else:
        return render_template('home.html')

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app)
#had to change from just simple app.run() to run_simple() due to threading issues since I am using lime. lime uses the main thread to run the app, so I had to use the run_simple() function to run the app on a different thread.

