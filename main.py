import pandas as pd
from streamlit.elements.arrow import Data
from data_analysis import DataAnalysis
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from evaluate_models import EvaluateModels
from data_cleaning_and_validation import DataCleaningAndValidation
import streamlit as st


# Project written by: Omar Moodie

# This project is split into 2 verions. 

# 1) Terminal

#     With the default terminal approach simply do a "python main.py" command in the terminal and it will run. 
#     Of course see the actually code in each .py file and choose different arguments. The terminal code is 
#     packed with more detail.

# 2) Streamlit

#     The streamlit portion of code was recommended to do by an instructor. It's in pretty good shape. In
#     the time limit that was given to complete the streamlit side of the project, the results turned out to 
#     be amazing. 





#
#
# 
# 
# 
# Below is streamlit portion of the code. Highlight it all, hold ctrl and press k then press u to uncomment. Re comment 
# again with ctrl and press k then press c. Of course only one section must be uncommented, while the other remains
# commented out, so the code runs.
#
# 
# To run below streamlit project, go to the terminal and type "streamlit run main.py"
# 
#





# Simple title for the starting page.
st.title('''
Cancer Diagnosis
#### Data Analysis, feature engineering, cleaning/validation and more are used to demonstrate data analysis on a cancer dataset.
''')

# A list will be made of class objects so the user will navigate step by step to the end.
if 'listIndex' not in st.session_state:
    st.session_state.listIndex = 0


# Definitely keep the dataframe in the session_state for reuse. There is a 
# copy of the dataframe called updatedDf because through a few classes, the 
# dataframe can be edited before it gets to the model building. 
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv('wisc_bc_data.csv')
    st.session_state.updatedDf = st.session_state.df.copy()


# 1 - Data Analysis - In all of data science, seeing the data clearly 
# is necessary and helps the engineers go on about the project. Since it
# shows data, pass it the dataframe to work with.
#
# 2 - Data Cleaning and Validation - A very necessary thing to do is clean
# and look after the data. Do missing values exist? If so, something must
# be done. 
#
# 3 - Feature engineering - Altering the features of a model are important
# because they can change the overall results like accuracy, precision, etc.
# Outliers can be handled here and if they are handled, various ways are
# possible.
#
# 4 - Model training - A lot of in depth model business going on such as 
# stacking, voting classifier, and more. The hyper parameters get optimized
# by Optuna which is a powerful hyperparameter tuning framework to increase
# results.
#
# 5 - Evaluate models - Probably my personal favorite part. Accuracy, recall,
# precision, f1 score and more are there to see model performance. 
#
#
# From feature engineering and onward, ONLY use the updatedDf in session_state.
if 'listOfClasses' not in st.session_state:
    # Each class will be in a list.
    st.session_state.listOfClasses = [DataAnalysis(None, True), 
                                    DataCleaningAndValidation(None, True), 
                                    FeatureEngineering(None, True), 
                                    ModelTraining(None, True), 
                                    EvaluateModels(None, None, None, True)]


# Page navigation button.
_ ,next = st.columns([10, 1])


# To ensure the button dissapears when the user reaches the last page, use an empty place holder.
place = st.empty()

if place.button('Next'):
    st.session_state.listOfClasses[st.session_state.listIndex].UpdateDataframe()
    st.session_state.listIndex += 1

    if st.session_state.listIndex + 1 >= len(st.session_state.listOfClasses):
        place.empty()

# All classes are derived/sub classes of the base class Data, which implements the Display function incase the streamlit approach is used.
st.session_state.listOfClasses[st.session_state.listIndex].Display()









#
#
# 
# 
# 
# Below is terminal code. Highlight it all, hold ctrl and press k then press u to uncomment. Re comment 
# again with ctrl and press k then press c. Of course only one section must be uncommented, while the other remains
# commented out, so the code runs.
#
#
# To run below terminal project, go to the terminal and type "python main.py"
#
#




# df = pd.read_csv('wisc_bc_data.csv')


# ''' Data Analysis - Just involves exploring the data for what is has. Class also has functions to 
#     calculate the PDF, CDF, and column/feature correlation depending on threshold.  '''
# da = DataAnalysis(df, False)
# da.ShowDataFrameDetails()
# da.ViolinPlot()
# da.CountPlot()
# da.BoxPlot()


# ''' Data Cleaning and Validating - Data sets is a normal process for a variety of reasons mentioned 
#     in the .py file. Outliers are the main responsibility given to this class and it has function to
#     REMOVE those outliers if desired. Take caution when removing outliers because that process can
#     potentially affect the resulting model in a negative fashion. '''
# dcav = DataCleaningAndValidation(df)
# dcav.IdentifyOutliersWithBoxPlot('radius_mean')
# # dcav.RemoveOutliers('radius_mean') 


# ''' Feature engineering - Performs operations on features to increase performance. Class holds 
#     functions that can display outliers with standard deviation, percentile, and calculate z score
#     as well. Variance inflation factor is most important and is done when the correlation function
#     is used. '''
# fe = FeatureEngineering(df)
# fe.LabelEncoding(True)
# fe.Correlation(2, True, True)
# # df = fe.OutliersPercentile('radius_mean', showSteps=False, removeOutliers=True)
# # df = fe.OutliersStandardDeviation('radius_mean', True, True)
# # fe.OutliersZScore()


# ''' Model training - Various types of models to be trained with TrainModelsVotingClassifier or 
#     TrainModelsStacking. All models are tuned well with Optuna, and newer models can easily be
#     added.'''
# mt = ModelTraining(df)
# mt.TrainModelsVotingClassifier(True)


# ''' Model metrics - There are various ways to measure how good a model is. Accuracy, precision, 
#     f1 score, and more. The interface in the EvaluateModels class handles this. '''
# em = EvaluateModels(mt.GetModels(), mt.GetXTest(), mt.GetYTest())
# em.BestModel('fp')