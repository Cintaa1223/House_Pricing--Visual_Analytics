import streamlit as st
import matplotlib.pyplot as plt
import webbrowser
from dataset_characteristics import X_frame,model,X_test
import shap

def init_shap1():
    shap.initjs() 
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values

def show_insights():
    explainer, shap_values = init_shap1()
    st.title("These are the insights gained from this project!")
    st.write("### Brief description of each plot:")
    st.write("- The waterfall plot shows which variables are the ones that affect the most the decision taken by the model to predict the target value (the contribution of every feature into the actual prediction with respect to the mean prediction). The blue values are the ones that ‘convince’ the model that the target value is 0; the red values ‘convince’ the model that the target value is 1.")
    st.write("- The force plot shows something very similar to the waterfall plot, but this time in terms of probability (if link = ‘logit’) or in terms of Shap value (without link = ‘logit’). In the end, the ‘side’ with higher probability (or higher absolute Shap value) is the one contributing the most to ‘convincing’ the model, and that will be the predicted value. Blue values contribute to the prediction of 0 and red values to the prediction of 1. The base value is the model’s probability prediction based on the probability (if link = ‘logit’) of predicting 1. In the case that there is no ‘logit’, a negative Shap base value predicts 0, and a positive one predicts 1.")
    st.write("- The decision plot shows the contribution of each individual variable in terms of probability to the decision of the model when having to predict. The decision plot’s straight vertical line marks the model’s base value. The colored line is the prediction. Feature values are printed next to the prediction line for reference. Starting at the bottom of the plot, the prediction line shows how the SHAP values (i.e., the feature effects) accumulate from the base value to arrive at the model’s final score at the top of the plot (final prediction).")

    # st.write("### How I believe the model works:")
    # st.write("Since many of the categorical variables represent the presence of some disease (or something bad), the variable having a value of 1 is something bad that could affect the fact of OF_PERSIS_ANYTIME=1, while the variable having a value of 0 would imply that the patient is healthy and, therefore, OF_PERSIS_ANYTIME=0. Therefore, if the model sees a variable has value of 1, this will contribute to the model’s prediction being 1, and vice versa")
    
    st.write('### Analysis of global force plot:')
    image1 = 'force_plot_logit.png'
    st.image(image1, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    image2 = 'force_plot_shap.png'
    st.image(image2, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    st.write("Both plots show the global explanation of the predictions of the model. Both have as x_axis the indices for the records used in the X_test dataset. However, the one with the link=’logit’ (at the top) has a y_axis showing the values for the probabilities, while the y_axis for the force plot without the link=’logit’ shows the Shap values. 0.0206 is the base value of the model as obtained using the SHAP Values. This means that if the total value is more than 0.0206, it signiﬁes that the house is more expensive than the median and if it is less than 0.0206, it signiﬁes that the house is cheaper than the median. The blue part of the graph pushes the prediction lower, and the red part is responsible for increasing it. This means that the instances in which there are a lot more red colored features will usually be 1 (being more expensive than the median) and vice versa.")
    st.write("For example, it is clear that for the first 150, the model’s prediction will be 1, while for the samples between indices 150 and 350, the model’s prediction will be 0. The analysis for the plot with link=’logit’ is similar. If the total probability value that appears when selecting an index in the plot is greater than 0.5, it signiﬁes that prediction is 1 and if it is less than 0.5, it signiﬁes that the prediction is 0.")

    st.write('### Analysis of global summary plot and global bar plot:')
    col1, col2 = st.columns([1, 1])
    with col1:
        f, axes = plt.subplots(1, 1)
        f.set_figheight(5)
        f.set_figwidth(15)
        shap_values=explainer(X_frame)
        fig = shap.summary_plot(shap_values, X_frame)
        st.pyplot(fig)
    with col2:
        f, axes = plt.subplots(1, 1)
        f.set_figheight(5)
        f.set_figwidth(15)
        fig2 = shap.plots.bar(shap_values)
        st.pyplot(fig2)
    st.write("Top 5 variables contributing more to the prediction: grade, yr_built, sqft_living, sqft_above, bathrooms. (Same ones for the top 5 variables that are contributing more to the average prediction value, as they are the same top 5 variables to be contributing the most in both the summary plot and in the bar plot).")
    st.write("The summary plot visualizes the eﬀects of the features on the prediction at diﬀerent values. The features at the top contribute more to the model than the bottom ones and thus have high feature importance. The color represents the value of the feature (blue meaning low, purple meaning the median value and red meaning high).")
    st.write("Grade: We can see that when the dots are blue (grade is low) its shap value is low too, and when it is red(grade is high) then its shap value is high, this means that they are positively correlated and having a good grade helps the house to be more expensive.")
    st.write("yr_built: We see now the oposite than grade, the value and shap_value are negatively correlated meaning the newer the house the cheaper the price.")
    st.write("The last 3 variables of our top 5 are also positively correlated and also have more density when the shap value is 0, meaning that it loses importance as the model cannot give a high or low shap value frequently. While the top 2 variables have densities in more important shap values, not 0.")
    

    st.write('### Analysis of year_built and sqft_living scatter plots using all shap values:')
    image5 = 'year_built_scatter_plot.png'
    st.image(image5, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    st.write("Here we can see furthermore the correlation between year built and its shap value, and how it is indeed negative, but we can see something that we noticed on the tableau section and it is that some new houses from 2015 and on have gotten high prices again and so its shap value may rise sometimes.")
    image6 = 'sqft_living_scatter_plot.png'
    st.image(image6, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
    st.write("Here we can see too the positive correlation between square footage and its shap value, we can see how until 2000 sqft almost all points are below 0 shap value and after 3000sqft almost all points are above 0 shap value, leaving the model 100sqft to struggle with.")
    if st.button('Go to Report'):
        webbrowser.open_new_tab('https://drive.google.com/file/d/1AzlQoihkLB0xlz132KDVjS8vbLvEGyuB/view?usp=sharing')