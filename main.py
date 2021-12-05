import pandas as pd
from data_analysis import DataAnalysis
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from evaluate_models import EvaluateModels

''' 

    This main file will be the main class to run other classes. 

'''

df = pd.read_csv('wisc_bc_data.csv')


# --- Data analysis just involves exploring the data for what is has. See data_analysis.py for explanation for last 3 default arguments.
da = DataAnalysis(df, True, True)


# --- Feature engineering performs operations on features to increase performance. Vital step.
fe = FeatureEngineering(df, True)
fe.EncodeCategoricalColumns()
fe.UpdateFeaturesForCorrelation()


# --- Model training is like letting a student study. Here a part of feature engineering is applied.  Return models as well.
mt = ModelTraining(df)
x_train, x_test, y_train, y_test = mt.GetTrainAndTestSets()
x_train, x_test = fe.UseScaling(x_train, x_test)
mt.Train(x_train, y_train, True)


# --- Use for accuracy and what not.
em = EvaluateModels(mt.GetModels(), x_test, y_test)
em.ViewClassificationReport()
em.ViewAccuracy()