import pandas as pd
from data_analysis import DataAnalysis
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from evaluate_models import EvaluateModels
import numpy as np


''' 

    This main file will be the main class to run other classes. 

'''


df = pd.read_csv('wisc_bc_data.csv')


# --- Data analysis just involves exploring the data for what is has. See data_analysis.py for explanation for last 3 default arguments.
da = DataAnalysis(df)
# da.ShowDataFrameDetails()
# da.ViolinPlot()
# da.CalculatePDF()
# da.CalculateCDF()
# da.HeatMap()
da.Correlation()


# t = ['hi', 5, 'd']

# while t:
#     print(t[0])
#     t.pop(0)






# # --- Feature engineering performs operations on features to increase performance. Vital step.
# fe = FeatureEngineering(df)
# fe.EncodeCategoricalColumns()
# fe.UpdateFeaturesForCorrelation()


# # --- Model training is like letting a student study. Here a part of feature engineering is applied.  Return models as well.
# mt = ModelTraining(df)
# x_train, x_test, y_train, y_test = mt.GetTrainAndTestSets()
# x_train, x_test = fe.UseScaling(x_train, x_test)
# mt.Train(x_train, y_train)
# mt.AdaBoostModel()
# mt.XgBoostModel()
# mt.GradientBoost()


# # --- Use for accuracy and what not.
# em = EvaluateModels(mt.GetModels(), x_test, y_test)
# em.ViewClassificationReport()
# em.ViewAccuracy()
# em.ViewConfusionMatrices()
# em.CustomModelMetrics()
# em.PlotRocCurves()
# em.ApplyRegularization(x_train, y_train)