import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings
filterwarnings('ignore')


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names


df, target_names = load_data()

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

st.sidebar.title('Input Features')
sepal_length = st.sidebar.slider('Sepal Length (cm)', float(
    df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider('Sepal Width (cm)', float(
    df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider('Petal Length (cm)', float(
    df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider('Petal Width (cm)', float(
    df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Prediction
prediction = model.predict(input_data)
prediction_species = target_names[prediction[0]]

st.write("Prediction")
st.write(f"The predicted species is: {prediction_species}")


# ------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier

# @st.cache_data
# def load_data():
#     iris = load_iris()
#     df = pd.DataFrame(iris.data, columns= iris.feature_names)
#     df['species'] = iris.target
#     return df, iris.target_names

# df, target_name = load_data()

# model = RandomForestClassifier()
# model.fit(df.iloc[:,:-1], df['species'])

# st.sidebar.title('input features')
# sepal_length = st.sidebar.slider('sepal length', float(df['sepal length cm'].min()), float(df['sepal length cm'].max()))
# sepal_width = st.sidebar.slider('sepal width', float(df['sepal width cm'].min()), float(df['sepal width cm'].max()))
# petal_length = st.sidebar.slider('petal length', float(df['petal length cm'].min()), float(df['petal length cm'].max()))
# petal_width = st.sidebar.slider('petal width', float(df['petal width cm'].min()), float(df['petal width cm'].max()))

# input_data = [[sepal_length,sepal_width,petal_length,petal_width]]

# #predition
# predition = model.predict(input_data)
# prediction_species = target_names[predition[0]]

# st.write("prediction")
# st.write(f"the prediction species is :{prediction_species}")
