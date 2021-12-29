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


''' ---Model training---

    Various types of models to be trained with any function with the following naming 
    convention: "TrainModels(RestOfNameHere)". All models are tuned well with Optuna, 
    and newer models can easily be added. '''
mt = ModelTraining(df)
mt.TrainModelsVotingClassifier()
# mt.TrainModelsStacking(True)


''' ---Model metrics---

    There are various ways to measure how good a model is. Accuracy, precision, f1 score,
    and more. The interface in the EvaluateModels class handles this. '''
em = EvaluateModels(mt.GetModels(), mt.GetXTest(), mt.GetYTest())
# em.PrecisionAndRecall()
# em.RocAucScore()
# em.F1Score()
# em.ClassificationReport()
# em.Accuracy()
# em.ConfusionMatrices()
# print('\n\nCustom Model Metrics:\n')
# em.CustomModelMetrics()
em.PlotRocCurves()