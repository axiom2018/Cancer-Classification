import pandas as pd
from streamlit.elements.arrow import Data
from data_analysis import DataAnalysis
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from evaluate_models import EvaluateModels
from data_cleaning_and_validation import DataCleaningAndValidation
import streamlit as st


st.title('''
Cancer Diagnosis
#### Data Analysis, feature engineering, cleaning/validation and more are used to demonstrate data analysis on a cancer dataset.
''')

if 'listIndex' not in st.session_state:
    st.session_state.listIndex = 0


# Definitely keep the dataframe in the session_state for reuse. There is a 
# copy of the dataframe called updatedDf because through each class, the 
# dataframe can be edited before it gets to the model building. Declaring it
# here and accessing it in other classes means for testing purposes the
# classes defined in "listOfClasses" can be switched around and still work fine.
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv('wisc_bc_data.csv')
    st.session_state.updatedDf = st.session_state.df.copy()


# 1 - In all of data science, seeing the data clearly is necessary
# and helps the engineers go on about the project. Since it shows
# data, pass it the dataframe to work with.
if 'listOfClasses' not in st.session_state:
    # Each class will be in a list.
    st.session_state.listOfClasses = [('feature_engineering', FeatureEngineering(st.session_state.df)),
        ('data_analysis', DataAnalysis(st.session_state.df)),
        ('data_cleaning_and_validation', DataCleaningAndValidation(st.session_state.df))
        ]


# Page navigation buttons
prev, _ ,next = st.columns([1, 10, 1])


# Managing events then the buttons are pressed.
if prev.button('Previous'):
    if st.session_state.listIndex <= 0:
        st.write('')
    else:
        st.session_state.listIndex -= 1

if next.button('Next'):
    if st.session_state.listIndex + 1 >= len(st.session_state.listOfClasses):
        st.write('')
    else:
        st.session_state.listOfClasses[st.session_state.listIndex][1].UpdateDataframe()
        st.session_state.listIndex += 1


# st.write(st.session_state.listOfClasses[st.session_state.listIndex][0])

st.session_state.listOfClasses[st.session_state.listIndex][1].Display()




# df = pd.read_csv('wisc_bc_data.csv')


# # --- Data analysis just involves exploring the data for what is has. See data_analysis.py for explanation for last 3 default arguments.
# da = DataAnalysis(df)
# # da.ShowDataFrameDetails()
# # da.ViolinPlot()
# # da.CalculatePDF()
# # da.CalculateCDF()
# # da.HeatMap()
# # da.Correlation()


# # --- Cleaning and validating data sets are a normal process for a variety of reasons mentioned in the .py file.
# dcav = DataCleaningAndValidation(df)
# # dcav.IdentifyOutliersWithBoxPlot('radius_mean')
# # dcav.HandleMissingValues()
# # Provide function with numerical columns from the dataset. For example, NOT "id" & "diagnosis". 
# # dcav.RemoveOutliers('texture_se') 





# # --- Feature engineering performs operations on features to increase performance. Vital step.
# fe = FeatureEngineering(df)
# fe.LabelEncoding()
# # df = fe.OutliersPercentile('radius_mean', showSteps=False, removeOutliers=True)
# # df = fe.OutliersStandardDeviation('radius_mean', True, True)
# # fe.OutliersZScore()

# # fe.Correlation(2, False, False)
# # fe.ShowHeatMap()
# # fe.Correlation()
# # fe.EncodeCategoricalColumns()
# # fe.UpdateFeaturesForCorrelation()


# ''' ---Model training---

#     Various types of models to be trained with any function with the following naming 
#     convention: "TrainModels(RestOfNameHere)". All models are tuned well with Optuna, 
#     and newer models can easily be added. '''
# mt = ModelTraining(df)
# mt.TrainModelsVotingClassifier()


# ''' ---Model metrics---

#     There are various ways to measure how good a model is. Accuracy, precision, f1 score,
#     and more. The interface in the EvaluateModels class handles this. '''
# em = EvaluateModels(mt.GetModels(), mt.GetXTest(), mt.GetYTest())
# em.BestModel('fp')