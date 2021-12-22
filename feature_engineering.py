import matplotlib.pyplot as plt
from numpy.core.fromnumeric import var
from pandas.core.indexes import category
from scipy.sparse import data
import seaborn as sns
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from feature_engine.encoding import MeanEncoder

from sklearn.model_selection import train_test_split

''' 

            Feature Engineering

Feature engineering is an extra few steps that can increase the overall model performance.

'''

class FeatureEngineering:
    def __init__(self, df):
        self.m_df = df

        # A part of feature engineering is getting rid of columns/features. The id column is useless.
        self.m_df.drop('id', axis=1, inplace=True)



    ''' Also the diagnosis column/feature should be encoded. No need for label encoding.
        It tends to be random, as in no correlation with the target variable. Mean encoding 
        however uses the target variable as the basis to make the new encoded feature. Its
        only slight drawback is potentially overfitting but that will be resolved soon. '''
    def MeanEncoding(self, showSteps=False):
        print('---Begin mean encoding.---')
        # Change boolean when testing to see head side by side with the result for comparison.
        seeDfHead = False

        if seeDfHead is True:
            print(self.m_df.head())

        if showSteps is True:
            print('Group by:')
            print(self.m_df.groupby(['diagnosis'])['radius_mean'].count())

            print('\nMean:')
            print(self.m_df.groupby(['diagnosis'])['radius_mean'].mean())

            print('\nMean Encoded:')

        me = self.m_df.groupby(['diagnosis'])['radius_mean'].mean().to_dict()
        self.m_df['diagnosis'] = self.m_df['diagnosis'].map(me)

        if showSteps is True:
            print(self.m_df)

        print('---Mean encoding finished.---\n')



    ''' This function helps the Correlation function. It will calculate the
        Vif scores for each column/feature in a dataframe and return the
        results.

        Simply create new columns in the new dataframe and in the vif_score
        feature/column, use list comprehension to GET the proper vif score.
        Also sort it to make it easier to see where the highest vif values are. '''
    def CalculateVifScore(self, df):
        vif = pd.DataFrame()
        vif['Features'] = df.columns
        vif['Vif_Score'] = [variance_inflation_factor(df.values, i) for i in range(self.m_df.shape[1])]
        vif.sort_values(by=['Vif_Score'], inplace=True, ascending=False)
        return vif




    ''' This function will handle all correlation activities in the dataset, because it's known
        that correlation can definitely affect models in negative ways.
        
        1) Heat map - Shows correlation in the dataset, but uses slicing because the whole dataset
            is much to be to be shown properly. 
            
        2) Vif - Variance inflation factor is a way to know which independent variables/columns
            are highly correlated in the data, so it's definitely useful. Luckily a library helps
            with this purpose. 
            
            To make things simplier, get a new dataframe and just map the vif numbers to each 
            column/feature. Every feature will have one anyway since the whole point of vif is
            to check a particular feature and see how correlated it is with every OTHER feature.
            
            threshold - The amount of highly correlated features to remove. Changing it will 
                definitely lead to a change in model performance.
                
            showHeatMap - Self explanatory.
            
            showSteps - There are a lot of steps that can be seen in the terminal to understand this
                function. Default value is false just to avoid blowing the terminal up and 
                getting to the point. '''
    def Correlation(self, threshold=2, showHeatMap=False, showSteps=True):
        if showHeatMap is True:
            plt.figure(figsize=(10, 10))
            matrix = sns.heatmap(self.m_df.iloc[:, 1:9].corr(), annot=True)
            plt.show()

        # Create the table of vif values and print it.
        vdf = self.CalculateVifScore(self.m_df)

        if showSteps is True:
            print(f'VIF results:\n{vdf}\n')

        ''' Get only the features column here. It will be used to get n amount of feature 
            names according to threshold. '''
        vdfFeatures = vdf['Features']

        if showSteps is True:
            print(f'Converting the vdf features to list:\n{vdfFeatures.to_list()}')
        featuresToLookInto = vdfFeatures.to_list()[:threshold]

        if showSteps is True:
            print(f'\nThe first {threshold} values according to threshold are: {featuresToLookInto}\n')


        ''' So based on the vif scores, what variables should be dropped and which
            should be KEPT? The correlation matrix is good to tell. With the features 
            look into gathered from the previous step, it's now necessary to see
            the correlation between all of those features and the target variable 
            'diagnosis'. For example, what's the correlation between radius_mean and
            diagnosis? How about perimeter_mean and diagnosis? Etc. 
            
            Start with the below variable highestValue. It will get the most highly 
            correlated column that correlates with the target value. '''
        highestValue = None

        for column in featuresToLookInto:
            # Get each specific column from dataframe and get correlation value.
            col1 = self.m_df['diagnosis']
            col2 = self.m_df[column]
            corrValue = round(col1.corr(col2), 5)

            if showSteps is True:
                print(f'Correlation between diagnosis and {column} is {corrValue}')

            ''' f the look runs for the first time, set the first value to being the
                highest. Otherwise make comparisons to see what value is the highest. '''
            if highestValue is None:
                highestValue = (column, corrValue)
            else:
                if corrValue > highestValue[1]:
                    highestValue = (column, corrValue)
                else:
                    continue

        if showSteps is True:
            print(f'\nHighest value of them all is: {highestValue}')
        
        if showSteps is True:
            print('Beginning process to remove columns.\n')

        ''' To loop backwards from list, since the highest value is at the end, one way
            of doing it is using a while loop while getting the highest index. For example
            if the code will remove 2 columns, 2 - 1 is 1. So the highest correlated 
            feature will be at index 1. '''
        i = len(featuresToLookInto) - 1
        while i >= 0:
            if showSteps is True:
                print(f'Removing feature/column: {featuresToLookInto[i]}.')
            
            # Drop the column and get the vif ratings again.
            self.m_df = self.m_df.drop([featuresToLookInto[i]], axis=1)

            # Create the new table of vif values and print it.
            vdf = self.CalculateVifScore(self.m_df)

            i -= 1

        if showSteps is True:
            print(f'-----Final VIF results table:-----\n{vdf}')