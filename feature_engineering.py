import matplotlib.pyplot as plt
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
    def MeanEncoding(self):
        # Change boolean when testing to see head side by side with the result for comparison.
        seeDfHead = False

        if seeDfHead is True:
            print(self.m_df.head())

        print('Group by:')
        print(self.m_df.groupby(['diagnosis'])['radius_mean'].count())

        print('\nMean:')
        print(self.m_df.groupby(['diagnosis'])['radius_mean'].mean())

        print('\nMean Encoded:')
        me = self.m_df.groupby(['diagnosis'])['radius_mean'].mean().to_dict()
        self.m_df['diagnosis'] = self.m_df['diagnosis'].map(me)
        print(self.m_df)



        ''' Error with the following code.
        
            Explanation: Following the library tutorial found here: 
            https://feature-engine.readthedocs.io/en/latest/encoding/MeanEncoder.html 
            
            The target column in the links dataset is survived. When using train_test_split,
            it is dropped from the actual dataset to give a different dataset to the function,
            of course the new one doesn't have target. Then the actual dataset gives the target
            value in the next argument. So this code follows the link example well in that 
            respect.  

            The links load_titanic function doesn't seem to do much. Not the replace function,
            the astype for the 'cabin' column, the astype for the 'pclass' column, nor the fillna
            should have anything to do with why the library won't work for THIS dataset.

            Things attempted:

            1) Dropping target variable 'diagnosis like link drops target variable (and more)
                in first argument of train_test_split:

                    Result error: "['diagnosis'] not found in axis"


            2) Changing 1st arg of train_test_split from "self.m_df.drop(['diagnosis'])" to
                self.m_df:

                    Result error: "Some of the variables are not categorical. Please cast
                    them as object or category before calling this transformer"

        '''

        # print('Using train test split.')
        # x_train, x_test, y_train, y_test = train_test_split(self.m_df, 
        #                                     self.m_df['diagnosis'], test_size=0.3, random_state=0)

        # print('Creating meanencoder object.')
        # encoder = MeanEncoder(variables=['diagnosis', 'radius_mean'])

        # encoder.fit(x_train, y_train)






    ''' This function will handle all correlation activities in the dataset, because it's known
        that correlation can definitely affect models in negative ways.
        
        1) Heat map - Shows correlation in the dataset, but uses slicing because the whole dataset
            is much to be to be shown properly. 
            
        2) Vif - Variance inflation factor is a way to know which independent variables/columns
            are highly correlated in the data, so it's definitely useful. Luckily a library helps
            with this purpose.  '''
    def Correlation(self, showHeatMap=False):
        if showHeatMap is True:
            plt.figure(figsize=(10, 10))
            matrix = sns.heatmap(self.m_df.iloc[:, 1:9].corr(), annot=True)
            plt.show()

        # ''' Create new dataframe and assign 2 columns. One will be all the column names
        #     of the dataframe passed in as an argument to the constructor. The other
        #     will be the VIF values themselves. Makes sense because every column/feature
        #     has a separate VIF score. '''
        # vDf = pd.DataFrame()

        # # Make a copy of the dataframe
        # dfCopy = self.m_df.copy()
        # dfCopy.drop(['id', 'diagnosis'], axis=1, inplace=True)

        # # Skip the first 2 columns, id and diagnosis. Get all other independent variables.

        # vDf['Features'] = dfCopy.columns
        # vDf['VIF_Value'] = [variance_inflation_factor(dfCopy.values, i) for i in range(dfCopy.shape[1])]
        # print(vDf)
