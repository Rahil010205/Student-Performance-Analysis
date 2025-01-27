from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

student_data = pd.read_csv('StudentPerformanceFactors.csv')
numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = student_data.drop('Exam_Score', axis=1)
y = student_data['Exam_Score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train_processed, y_train)

y_test_lr_pred = linear_regressor.predict(X_test_processed)

def predict_student_performance(numerical_features, categorical_features):
    user_input = {}
    user_input['Hours_Studied'] = float(numerical_features[0])
    user_input['Attendance'] = float(numerical_features[1])
    user_input['Sleep_Hours'] = float(numerical_features[2])
    user_input['Previous_Scores'] = float(numerical_features[3])
    user_input['Tutoring_Sessions'] = float(numerical_features[4])
    user_input['Physical_Activity'] = float(numerical_features[5])

    user_input['Parental_Involvement'] = (categorical_features[0])
    user_input['Access_to_Resources'] = (categorical_features[1])
    user_input['Extracurricular_Activities'] = (categorical_features[2])

    user_data = pd.DataFrame([user_input])
    user_data_processed = preprocessor.transform(user_data)
    predicted_score_lr = linear_regressor.predict(user_data_processed)
    return predicted_score_lr[0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    question1 = request.form['question1']
    question2 = request.form['question2']
    question3 = request.form['question3']
    question4 = request.form['question4']
    question5 = request.form['question5']
    question6 = request.form['question6']

    question7 = request.form['question7'] 
    question10 = request.form['question8']
    question13 = request.form['question9']
  
    r = predict_student_performance([question1, question2, question3, question4, question5, question6], [question7, question10, question13])
    r = round(r,2)
    return render_template('index.html', results=r) 

if __name__ == '__main__':
    app.run(debug=True)
