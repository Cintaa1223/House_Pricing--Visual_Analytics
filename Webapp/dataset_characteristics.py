import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

import shap

#To turn of the warnings for better visualization
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def load_data(filename):
    data = pd.read_csv(filename, sep=",")
    data = data.drop('Unnamed: 0', axis = 1)
    return data

init_data = load_data("init_data.csv")
binary_data = load_data("data.csv")

def plot_seaborn(data1, data2, col, title1, title2, title, color1, color2):
    f, axes = plt.subplots(1, 3)
    f.set_figheight(5)
    f.set_figwidth(15)
    
    sns.distplot(data1[col], ax=axes[0], color=color1)
    axes[0].set_title(title1)
    
    sns.distplot(data2[col], ax=axes[1], color=color2)
    axes[1].set_title(title2)
    
    sns.distplot(data1[col], label=title1, ax=axes[2], color=color1)
    sns.distplot(data2[col], label=title2, ax=axes[2], color=color2)
    axes[2].set_title(title)
    plt.legend()

    return f

def boxplot_seaborn(male, female, col, title1, title2, title, color1, color2):
    f, axes = plt.subplots(1, 3)
    f.set_figheight(5)
    f.set_figwidth(15)
    
    sns.boxplot(male[col], ax=axes[0], color=color1, orient='v')
    axes[0].set_title(title1)
    
    sns.boxplot(female[col], ax=axes[1], color=color2, orient='v')
    axes[1].set_title(title2)
    
    sns.boxplot(male[col], ax=axes[2], color=color1, orient='v')
    sns.boxplot(female[col], ax=axes[2], color=color2, orient='v')
    axes[2].set_title(title)
    
    return f

def split_dataset(data2):
    X_frame = data2.iloc[:,data2.columns != 'price']
    y = data2.iloc[:, data2.columns == 'price']
    X_train, X_test, y_train, y_test = train_test_split(X_frame, y, test_size=0.2, random_state=42)
    return X_frame, X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    return model, predictions, y_pred

X_frame, X_train, X_test, y_train, y_test = split_dataset(binary_data)
model, predictions, y_pred = train_model(X_train, y_train, X_test)

def init_shap1():
    shap.initjs() 
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values

def plot_shap_global(explainer, shap_values):
    st.write("""### Force Plot Using SHAP Library""")
    im1 = 'force_plot_logit.png'
    im2 = 'force_plot_shap.png'
    st.image(im1, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    st.image(im2, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    st.write("""### Summary Plot Using SHAP Library""")
    f, axes = plt.subplots(1, 1)
    f.set_figheight(5)
    f.set_figwidth(15)
    shap_values=explainer(X_frame)
    fig = shap.summary_plot(shap_values, X_frame)
    st.pyplot(fig)
    # im3 = 'global_summary_plot.png'  
    # st.image(im3, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    st.write("""### Bar Plot Using SHAP Library""")
    # im4 = 'global_bar_plot.png'
    # st.image(im4, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    f, axes = plt.subplots(1, 1)
    f.set_figheight(5)
    f.set_figwidth(15)
    fig2 = shap.plots.bar(shap_values)
    st.pyplot(fig2)

    st.write("""### Scatter plot of Shap values for Year Built""")
    im5 = 'year_built_scatter_plot.png'
    st.image(im5, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    st.write("""### Scatter plot of Shap values for Square foot Living""")
    im6 = 'sqft_living_scatter_plot.png'
    st.image(im6, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    # st.write("""### Scatter plot of Shap values for Number of Floors""")
    # im7 = 'floors_scatter_plot.png'
    # st.image(im7, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

    st.write('You can find all these Shap plots deeply explained in the Insights section.')

def show_characteristics_page():
    st.title("Explainable AI & Visualization Webapp for House Pricing")

    st.write("""### This is the House Pricing dataset""")
    st.dataframe(init_data)

    dataLowGrade = binary_data[binary_data['grade']<=7]
    dataHighGrade = binary_data[binary_data['grade']>7]

    st.write("""### These are the distributions of the main features (Low Grade vs. High Grade)""")
    F1 = plot_seaborn(dataLowGrade, dataHighGrade, 'floors', 'Floors Low Grade', 'Floors High Grade',  'Floors Distribution', 'GREEN', 'RED')
    S1 = plot_seaborn(dataLowGrade, dataHighGrade, 'sqft_living', 'Square foot Living Low Grade', 'Square foot Living High Grade',  'Square foot Living Distribution', 'GREEN', 'RED')
    P1 = plot_seaborn(dataLowGrade, dataHighGrade, 'price', 'Price Low Grade', 'Price High Grade',  'Price Distribution', 'GREEN', 'RED')
    
    st.pyplot(F1)
    st.pyplot(S1)
    st.pyplot(P1)


    # st.write("""### These are the boxplots of the main features (Low Grade vs. High Grade)""")
    # F2 = boxplot_seaborn(dataLowGrade, dataHighGrade, 'floors', 'Floors Low Grade', 'Floors High Grade',  'Floors Distribution', 'AQUAMARINE', 'PLUM')
    # S2 = boxplot_seaborn(dataLowGrade, dataHighGrade, 'sqft_living', 'Squarefoot Living Low Grade', 'Squarefoot Living High Grade',  'Squarefoot Living Distribution', 'AQUAMARINE', 'PLUM')
    # P2 = boxplot_seaborn(dataLowGrade, dataHighGrade, 'price', 'Price Low Grade', 'Price High Grade',  'Price Distribution', 'AQUAMARINE', 'PLUM')
    
    # st.pyplot(F2)
    # st.pyplot(S2)
    # st.pyplot(P2)

    st.write("""### This is the heatmap with the correlation values for each variable in the matrix""")
    image1 = 'heatmap.png'
    st.image(image1, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    
    st.write("""### This is the confusion matrix obtained from the predictions, after training the model, and its characteristics""")
    conf_mat = confusion_matrix(y_test, y_pred)
    st.write(conf_mat)
    PRA = classification_report(y_test, y_pred, output_dict = True)
    st.dataframe(PRA)

    st.write("""### This is the density chart of the probabilities of the prediction""")
    probs = model.predict_proba(X_test)
    prob_1 = [prob[1] for prob in probs]
    plot_df = pd.DataFrame({'Target': y_test.squeeze(), 'Prob': prob_1})
    plot_df0 = plot_df[plot_df['Target']==0]
    plot_df1 = plot_df[plot_df['Target']==1]
    density_chart = plot_seaborn(plot_df0, plot_df1, 'Prob', 'Class: 0', 'Class: 1',  'Density Chart', 'RED', 'GREEN')
    st.pyplot(density_chart)

    st.write("### A global interpretation using all shap values")
    explainer, shap_values = init_shap1()
    plot_shap_global(explainer, shap_values)