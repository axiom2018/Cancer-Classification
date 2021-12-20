import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


''' 

            Data Cleaning and Validation

Things such as fixing structural errors, selecting which observations to eliminate, or handling missing data and more
are what make up data cleaning and validation. 

'''

class DataCleaningAndValidation:
    def __init__(self, df, showColumnNames=False, showNullValues=False):
        self.m_df = df

        ''' The user being able to visually see the column names will be a help
            to them with this class interface. '''
        if showColumnNames is True:
            print(self.m_df.columns)

        ''' Show any empty or null/missing values. Can also be done with dataframeObject.isna().any() which
            returns booleans if null values exist for a feature/column. '''
        if showNullValues is True:
            print(self.m_df.isnull().sum())

    

    ''' Outliers are just those bits of data that are more unique, that stray away from
        the bulk. They can definitely have some effect on the model. 
        
        But beware, there must be a legitimate reason to get rid of an outlier. If so, the
        model performance will improve. Don't get rid of one just because it's numrical 
        value might be bigger than expected. The model could probably benefit from that.
        
        To SEE outliers, it's possible to use Boxplots. They have the ability to show 
        outliers, so they're useful. '''
    def IdentifyOutliersWithBoxPlot(self, columnName):
        # Check if the argument given is in fact a feature/column.
        if columnName not in self.m_df.columns.to_list():
            print('Bad column name given, returning.')
            return

        # The potential outliers will be given based on the column/feature argument provided.
        sns.boxplot(x=columnName, data=self.m_df)
        plt.show()



    ''' Remember that removing outliers can potentially be costly. Using functions like
        "drop" isn't very ideal because that will eliminate an entire row of potentially
        valuable information. If the dataset is really big and maybe a few rows might
        be removed due to the "drop" function, that might be an acceptable case to use it.

        But the cancer dataset is not that big and also is very imbalanced. So using
        dropna isn't ideal. '''
    def RemoveOutliers(self, columnName):
         # Check if the argument given is in fact a feature/column.
        if columnName not in self.m_df.columns.to_list():
            print('Bad column name given, returning.')
            return
            

        ''' Important values to be calculated.
            q1 - # between smallest value and median.
            q3 - # between median and highest value.
            iqr - interquartile range. 25th to 75th percentile.
        '''
        sns.boxplot(x=columnName, data=self.m_df)
        plt.show()



    ''' Handling any missing values in the dataset is a big part of data cleaning.
        Simple to do as well. As mentioned in the comment right above the previous
        function, using drop isn't ideal. Neither is imputing. See the following:
        https://elitedatascience.com/data-cleaning . 
        
        With the link above in mind, if a column/feature is categorical or numerical,
        it will be handled differently.
        
        The dummy example is there to show how to deal with missing values if none
        are present in the projects default dataset. '''
    def HandleMissingValues(self, dummyExample=True):
        if dummyExample is True:
            # Create dummy dataset.
            print('Created dummy dataset.\n')

            ''' Types that Dataframes work with:
                1) 1 to n - int64
                2) Decimal values - float64
                3) Strings - object '''
            testDf = pd.DataFrame(data={ 
                "City":['Los Angeles', 'Seattle', pd.NA, 'Miami', 'Hollywood (CA)', 'Manhattan', 'New Orleans', pd.NA],
                "Square Feet":[np.nan, 1122, 895, 905, 800, np.nan, 855, 985],
                "Average Rent":[1200, np.nan, 1350, 900, np.nan, 975, 850, 1400]
            })

            print(f'Dataset pre cleaing:\n{testDf}')

            ''' Then as link suggested, get rid of categorical columns with 
                'Missing' string, and replace the numerical columns with 0. '''
            testDf['City'].replace(pd.NA, 'Missing', inplace=True)


            print('Square feet null values:')
            print(testDf['Square Feet'].isnull().sum())

            print('Average Rent null values:')
            print(testDf['Average Rent'].isnull().sum())

            testDf['Square Feet'].fillna(0, inplace=True)
            testDf['Average Rent'].fillna(0, inplace=True)
            print(f'Final test dataset after cleaning:\n{testDf}')
