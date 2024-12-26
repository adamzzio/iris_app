import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

st.write("""
# Simple Iris Flower Web App
## Iris Flower Prediction
This app predicts the **Iris flower** type and also can do clustering!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write("""
## Iris Flower Clustering Based On Sepal Length and Petal Length
""")

st.sidebar.header('User Input Parameters')

def user_input_clusters():
    n = st.sidebar.slider('N of clusters', 1, 8, 3)
    return n

jumlah = user_input_clusters()

dct = {"N of Clusters":[jumlah]}
df1 = pd.DataFrame(dct)

st.subheader('User Input parameters')
st.write(df1)

dt = pd.DataFrame(X)

kmeans = KMeans(n_clusters = jumlah, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(dt[[0,2]])

st.subheader('The Clustering Result')
st.write(y_kmeans)

LABEL_COLOR_MAP = {0:'red', 1:'blue', 2:'green', 3:'black', 4:'yellow', 5:'violet', 6:'orange', 7:'cyan', 8:'magenta'}
label_color = [LABEL_COLOR_MAP[l] for l in y_kmeans]
xs = dt[[0]]
ys = dt[[2]]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plot = plt.scatter(xs, ys, c=label_color)
st.write(fig)

st.write("""
# Author : Adam Maurizio Winata
# Source Ideas : FreeCodeCamp Youtube Channel
""")
