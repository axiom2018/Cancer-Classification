import pandas as pd
from data_analysis import DataAnalysis
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from evaluate_models import EvaluateModels
from data_cleaning_and_validation import DataCleaningAndValidation
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
# da.Correlation()


# --- Cleaning and validating data sets are a normal process for a variety of reasons mentioned in the .py file.
dcav = DataCleaningAndValidation(df)
# dcav.IdentifyOutliersWithBoxPlot('radius_mean')
# dcav.HandleMissingValues()
# Provide function with numerical columns from the dataset. For example, NOT "id" & "diagnosis". 
# dcav.RemoveOutliers('texture_se') 





# --- Feature engineering performs operations on features to increase performance. Vital step.
fe = FeatureEngineering(df)
fe.LabelEncoding()
# df = fe.OutliersPercentile('radius_mean', showSteps=False, removeOutliers=True)
# df = fe.OutliersStandardDeviation('radius_mean', True, True)
# fe.OutliersZScore()

# fe.Correlation(2, False, False)
# fe.ShowHeatMap()
# fe.Correlation()
# fe.EncodeCategoricalColumns()
# fe.UpdateFeaturesForCorrelation()


# --- Model training is like letting a student study. Here a part of feature engineering is applied.  Return models as well.
mt = ModelTraining(df)
# mt.ApplyLogisticRegression()
# mt.ApplyRandomForest()
# mt.ApplyDecisionTree()
# mt.TrainModelsVotingClassifier(True)
mt.TrainModelsStacking(True)

# test = ('Test', object, 89.5)
# print(test)
# print(test[2])

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