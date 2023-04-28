import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import shap

from add_house_data import new_houses, init_data, binary_data

median_price = init_data['price'].describe()['50%']

def load_model():
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

model = load_model()

def pred_new_house():
    idx = 0
    st.write("You chose to do the prediction for a new house.")
    if len(new_houses.keys()) == 0:
        st.write("Add a new house first!")
        return 0
    options = []
    for val in new_houses.keys():
        options.append(val)
    house = st.selectbox("Which new house do you want to predict for?", set(options))
    st.write("If you want to predict for a new house that has not been added yet, go first to 'Add House Data' and add it.")
    ok_new = st.button("Predict for new house!")
    if ok_new:
        idx, X_frame2 = predict_target(house)
        plot_shap(new_houses[house], X_frame2)


def pred_existing_house():
    idx = 0
    st.write("You chose to do the prediction for an already existing house.")
    index = st.slider('Insert the ID of the house you want to predict for', 0, len(init_data)-1)
    ok_existing = st.button("Predict for existing patient!")
    if ok_existing:
        idx, X_frame2 = predict_target(index)
        plot_shap(idx, X_frame2)


def predict_target(house):
    idx = house
    if isinstance(house, str):
        idx = new_houses[house]
        st.write("This is the prediction for the house with ID ",house,"!")
    else:
        st.write("This is the prediction for the chosen house!")
    X_frame2 = binary_data.iloc[:, binary_data.columns != 'price']
    y_pred2 = model.predict(X_frame2)
    y2 = binary_data.iloc[:, binary_data.columns == 'price'].to_numpy()
    
    if y_pred2[idx] == 0:
        st.write('The prediction of the classification for this house is ', y_pred2[idx],', meaning that its predicted price in USD is <= ', median_price,'.')
    else:
        st.write('The prediction of the classification for this house is ', y_pred2[idx],', meaning that its predicted price in USD is > ', median_price, '.')
    
    if y2[idx][0] == 0:
        st.write('The real target of the house was ', y2[idx][0],', meaning that its price in USD was lower than ',median_price,'.')
    else:
        st.write('The real target of the house was ', y2[idx][0],', meaning that its price in USD was higher than ',median_price,'.')
    return idx, X_frame2

def init_shap2(X_frame2):
    shap.initjs()
    explainer = shap.Explainer(model)
    shap_values = explainer(X_frame2)
    return explainer, shap_values


def plot_shap(idx, X_frame2):
    explainer, shap_values = init_shap2(X_frame2)
    st.write("""### Waterfall Using SHAP Library""")
    fig1, ax = plt.subplots(1, 1)
    shap.plots.waterfall(shap_values[idx])
    st.pyplot(fig1)
    st.write("""### Decision Plot Using SHAP Library""")
    fig3, ax = plt.subplots(1, 1)
    shap.decision_plot(explainer.expected_value, shap_values[idx].values, X_frame2.iloc[idx], link = 'logit')
    st.pyplot(fig3)
    st.write("A global interpretation using all shap values can be found in the 'Characteristics' page.")


def show_prediction_page():
    st.title("House Price Prediction")
    st.write("""### Choose a house to do the predictrion for!""")    

    st.write("### What type of house do you want to do the prediction for?")
    pred = st.selectbox('New House or Existing House', ("New added house", "Already existing house"))
    if pred == "New added house":
        pred_new_house()
    else:
        pred_existing_house()