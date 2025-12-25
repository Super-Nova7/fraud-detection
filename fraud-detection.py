import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import joblib

#dataset
df = pd.read_csv("creditcard.csv")

scaled = ["Time","Amount"]

#encoded and scaling the data

preprocess = ColumnTransformer(transformers=[("scale",StandardScaler(),scaled)],remainder="passthrough")

#use of pipeline
pipeline = Pipeline(steps=[("preprocessing",preprocess),("model",LogisticRegression(class_weight="balanced",max_iter=100))])

#defining target and feature column
X = df.drop("Class",axis=1)
Y = df["Class"]

#splitting data for training and testing
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

#training the model
pipeline.fit(xtrain,ytrain)

#making predictions
prediction_1 = pipeline.predict(xtest)


#evaluating model by metrics
print(confusion_matrix(ytest,prediction_1))

#use of threshold
chances = pipeline.predict_proba(xtest)[:,1]
threshold = 0.3
prob = (chances >= threshold).astype(int)
print(confusion_matrix(ytest,prob))


#use of precision_recall_curve to find the best threshold value which is 0.5(default)
precision,recall,threshold= precision_recall_curve(ytest,chances)

#saving the model
joblib.dump(pipeline,"fraud-detection.pkl")



