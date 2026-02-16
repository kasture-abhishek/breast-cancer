import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

import plotly.express as px
from sklearn.metrics import accuracy_score

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score
)
df=pd.read_csv("breast-cancer.csv")
df.head()
df.columns
df.describe().T.style.background_gradient(cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True))
##px.histogram(data_frame=df, x="smoothness_mean",color="diagnosis",color_discrete_sequence=["#A865C9","#f6abb6"])

px.histogram(data_frame=df,x="smoothness_mean",color="diagnosis",color_discrete_sequence=["#A865C9","#f6abb6"])
px.histogram(data_frame=df,x="texture_mean",color="diagnosis",color_discrete_sequence=["#A865C9","#f6abb6"])
px.scatter(data_frame=df,x="symmetry_worst",color="diagnosis",color_discrete_sequence=['#A865C9','#f6abb6'])
px.scatter(data_frame=df, x="concavity_worst",color="diagnosis",color_discrete_sequence=['#A865C9','#f6abb6'])
px.scatter(data_frame=df, x="fractal_dimension_worst",color="diagnosis",color_discrete_sequence=['#A865C9','#f6abb6'])
##Data Preprocessing

df.columns

#df.drop("id",axis=1,inplace=True)
df["diagnosis"]=(df["diagnosis"]=="M").astype(int)
#encode the label into 1/0
df.head()
corr=df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr,cmap=sns.color_palette("ch:s=-.2,r=.6",as_cmap=True),annot=True)
plt.show()
# Get the absolute value of the correlation
corr_target=abs(corr["diagnosis"])
#Select highly correlated features(thresold=0.2)
relevant_features=corr_target[corr_target>0.2]
names=[index for index,value in relevant_features.items()]
names.remove("diagnosis")
X=df[names]
y=df['diagnosis']
#Split the data in to traning and validation
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#create an instance of standard scaler
scaler=StandardScaler()
#fit it in the training data
scaler.fit(X_train)
#transform training data 
scaler.transform(X_train)
#transform validating data
scaler.transform(X_test)

#Model Evalution

def train_evaluate_model(model, X_train,y_train,X_test,y_test):
    '''
    Keyword arguments:
    X--Training data
    y--Training labels
    returns a dataframe for evaluating metrics
    '''
    model.fit(X_train,y_train)#fit the model instance
    predictions=model.predict(X_test)#alculate predictions
    print(predictions)
    #compute metrics for evalution
    accuracy=accuracy_score(y_test,predictions)
    f1=f1_score(y_test,predictions)
    precision=precision_score(y_test,predictions)
    recall=recall_score(y_test,predictions)
    balanced_accuracy=balanced_accuracy_score(y_test,predictions)
    #create a dataframe to visualize the results
    eval_df=pd.DataFrame([[accuracy,f1,precision,recall,balanced_accuracy]],columns=['accuracy','f1_score','precision','recall','balanced_accuracy'])
    return eval_df

lg=LogisticRegression()
results=train_evaluate_model(lg,X_train,y_train,X_test,y_test)
results.index=["LogisticRegression"]
results.sort_values(by="f1_score",ascending=False).style.background_gradient(cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True))
#evaluation
models=results.T.columns.values
models=models[0:7]
models
fig=px.bar(x=results.iloc[:7,0].values , y=models,orientation="h",color=results["accuracy"].iloc[0:7],color_continuous_scale="tealrose",template="seaborn")
fig.update_layout(width=500,height=250,xaxis=dict(title="Accuracy"),
                                       yaxis=dict(title="Models"))
fig.show()


fig=px.bar(x=results.iloc[:7,0].values , y=models,orientation="h",color=results["f1_score"].iloc[0:7],color_continuous_scale="bupu",template="seaborn")
fig.update_layout(width=500,height=250,xaxis=dict(title="F1_Score"),
                                       yaxis=dict(title="Models"))
fig.show()


fig=px.bar(x=results.iloc[:7,0].values , y=models,orientation="h",color=results["precision"].iloc[0:7],color_continuous_scale="sunsetdark",template="seaborn")
fig.update_layout(width=500,height=250,xaxis=dict(title="Precision"),
                                       yaxis=dict(title="Models"))
fig.show()


fig=px.bar(x=results.iloc[:7,0].values , y=models,orientation="h",color=results["recall"].iloc[0:7],color_continuous_scale="algae",template="seaborn")
fig.update_layout(width=500,height=250,xaxis=dict(title="Recall"),
                                       yaxis=dict(title="Models"))
fig.show()


fig=px.bar(x=results.iloc[:7,0].values , y=models,orientation="h",color=results["balanced_accuracy"].iloc[0:7],color_continuous_scale="solar",template="seaborn")
fig.update_layout(width=500,height=250,xaxis=dict(title="Balanced-Accuracy"),
                                       yaxis=dict(title="Models"))
fig.show()