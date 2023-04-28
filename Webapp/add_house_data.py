import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from dataset_characteristics import init_data, binary_data

new_houses = {} #Stores the index values of the new houses added to the dataset
median_price = init_data['price'].describe()['50%']

def enter_values(init_data, binary_data):
    features = set_features_default(init_data)
    houseID = st.text_input('ID of the house:')
    
    with st.form("my_form"):
        
        features['price'] = st.number_input('What is the house price in USD?')
        features['bedrooms'] = st.slider('How many bedrooms does the house have?',0,int(init_data['bedrooms'].max()))
        features['bathrooms'] = st.slider('How many bathrooms does the house have?',0.0,init_data['bathrooms'].max(),0.25)
        features['sqft_living'] = st.number_input('What is the size of the living room in square foot?')
        features['floors'] = st.slider('How many floors does the house have?',0.0,init_data['floors'].max(),0.5)
        st.write('Does the house have...?')
        temp = st.checkbox('Waterfront')
        if temp:
            features['waterfront'] = 1
        else:
            features['waterfront'] = 0
        features['condition'] = st.slider('How would you grade the condition of the house?(1=bad, 5=great)',1,5)
        features['grade'] = st.slider('Have some users graded your house? Which score does it have between 1 and 13:',1,13)
        features['sqft_above'] = st.number_input('What is the size of the loft in square foot?')
        features['sqft_basement'] = st.number_input('What is the size of the basement in square foot?')
        features['yr_built'] = st.slider('When was the house built?',init_data['yr_built'].min(),init_data['yr_built'].max())

        
        
        st.write("Would you like to submit the introduced values?")
        submitted = st.form_submit_button("Submit")

            
    if submitted:
        st.write(features['price'])
        if houseID:
            st.write('The ID of the house you have just added is ', houseID, '.')
            insert_features_into_dataset(features, init_data, houseID)
            features2 = set_features2(features)            
            insert_features_into_dataset(features2, binary_data, houseID)
            st.write("If you want to add a new house, enter the ID of the new house and the new values for each feature, and press submit again.")
        else:
            st.write("Enter first the ID of the house!")

def set_features2(x):
    features2 = x
    if x['price'] <= median_price:
        features2['price'] = 0
    else:
        features2['price'] = 1
    return features2

def set_features_default(data2):
    features = {}
    st.write("All values are set to the minimum value by default.")
    st.write("Feel free to change the value for those features!")
    for col in data2.columns:
        features[col] = int(0)
    features['price'] = float(features['price'])
    return features

# @st.cache
def insert_features_into_dataset(features, data2, houseID):
    new_houses[houseID] = len(data2)
    data2.loc[len(data2)] = features

@st.cache
def insert_features_into_dataset2(features, data2, patient):
    data2.loc[len(data2)] = features
    # X_test2.loc[len(X_test2)] = data2.iloc[-1].drop('OF_PERSIS_ANYTIME')



def add_house_page():
    st.title("Add new house")

    enter_values(init_data, binary_data)
    # dfs = [X_test, X_test2]
    # updated = pd.concat(dfs, axis=0, ignore_index=True) 

    st.write("You can check in the dataset below how the new house has been successfully added!")
    st.dataframe(init_data)
    st.dataframe(binary_data)