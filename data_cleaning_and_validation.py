import matplotlib.pyplot as plt
from pandas.core import frame
import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sp
import streamlit as st
from data import Data

''' 

            Data Cleaning and Validation

Things such as fixing structural errors, selecting which observations to eliminate, or handling missing data and more
are what make up data cleaning and validation. 

'''

class DataCleaningAndValidation(Data):
    ''' Python only allows for 1 constructor unfortunately. So the extra "streamLitInit" default argument
        means the class is being instantiated with the use of streamlit in mind. Streamlit uses session states
        to manage variables so the usual "self" really isn't necessary. The functions that are on the class,
        that will be called when the streamlit code in main.py is ran, are Display & UpdateDataframe. '''
    def __init__(self, df, streamLitInit=False):
        if streamLitInit is False:
            self.m_df = df
            self.m_checkBoxClicked = False



    # Streamlit function Display is overriding base class function in Data.py
    def Display(self):
        # Get the dataframe through session state to carry out rest of function code.
        updatedDf = st.session_state.updatedDf
        
        st.write('### [Data Cleaning and Validation] Cleaning the data is ALWAYS a big part of the job.')
        st.write('##### Outlier removal gets rid of extra data points that might not belong.')

        # Total column length is 32 for the record. No need for diagnosis nor id for now. 
        st.write('')
        st.write('')
        columnName = st.selectbox("Select column to see its outliers:", (list(updatedDf.columns[2:])))

        
        fig = plt.figure(figsize=(10, 6))


        ''' The boolean value will help controlling what's displayed, especially for the
            if statements below. '''
        if st.checkbox('Remove outliers (If desired, it not, leave unchecked & click next)'):
            self.m_checkBoxClicked = True
        else:
            self.m_checkBoxClicked = False


        ''' Since removing outliers can be a bit dangerous, it's definitely needed
            to do the operations in a copy first. Even if a copy IS made, if the
            checkBoxClicked boolean isn't true, the changes won't be shown of course.
            
            boxplot - Definitely use box plots to show the outliers.
            
            st.pyplot - Shows the above boxplot.
            
            DataframeDetails - This will show some changes in the dataframe
                IF the user decided to removal outliers.
            
            updatedDf - If outliers are removed and the user decides to carry
                on with them, this dataframe will hold those changes.  '''
        if 'dfCopy' not in st.session_state and self.m_checkBoxClicked is False:
            sns.boxplot(x=columnName, data=updatedDf)
            st.pyplot(fig)

            # Show details so user can see more regarding changes.
            self.DataframeDetails(updatedDf)
        
        elif 'dfCopy' not in st.session_state and self.m_checkBoxClicked is True:
            st.session_state.dfCopy = self.RemoveOutliers(columnName, False, True)
            sns.boxplot(x=columnName, data=st.session_state.dfCopy)
            st.pyplot(fig)
    
            self.DataframeDetails(st.session_state.dfCopy)

        elif 'dfCopy' in st.session_state and self.m_checkBoxClicked is False:
            sns.boxplot(x=columnName, data=updatedDf)
            st.pyplot(fig)

            self.DataframeDetails(updatedDf)
        
        elif 'dfCopy' in st.session_state and self.m_checkBoxClicked is True:
            st.session_state.dfCopy = self.RemoveOutliers(columnName, False, True)
            sns.boxplot(x=columnName, data=st.session_state.dfCopy)
            st.pyplot(fig)
    
            self.DataframeDetails(st.session_state.dfCopy)

    # Streamlit function UpdateDataframe is overriding base class function in Data.py
    def UpdateDataframe(self):
        if self.m_checkBoxClicked is True:
            st.session_state.updatedDf = st.session_state.dfCopy
            print(f'User decided to keep the changes made. Df shape: {st.session_state.updatedDf.shape}')
        else:
            print(f'User decided NOT to use/keep any outlier changes. Df shape: {st.session_state.updatedDf.shape}')
    

    # Helper function of the above streamlit overriden function Display.
    def DataframeDetails(self, df):
        st.write('### Dataframe details:')
        st.write(f'Dataframe shape (Rows/Columns): {df.shape}')
        st.write('Datframe: ')
        st.write(df.head(10))



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
        dropna isn't ideal.
        
        The IQR or inter-quartile range will be used to remove the outliers. Outliers
        are outliers because they are a distance away from the normal points, hence why
        calculating an upper and lower boundary are necessary. Box plot is used to visualize
        the request to remove outliers from a particular feature/column.

        Due to the streamlit request, the proper dataframe must handled in this function.
        There's one being used by streamlit that's saved in the session state, and of
        course a different one passed into the first constructor of this class. Both
        purposes need to be handled, hence the if statement at the beginning of this 
        function.   '''
    def RemoveOutliers(self, columnName, describe=False, streamLitRequest=False):
        frameCopy = None
        if streamLitRequest is True:
            frameCopy = st.session_state.updatedDf.copy()
        else:
            frameCopy = self.m_df.copy()

         # Check if the argument given is in fact a feature/column.
        if columnName not in frameCopy.columns.to_list():
            print('Bad column name given, returning.')
            return

        if streamLitRequest is False:
            print(f'Currently working with column {columnName}.')

        # Show stats since some of its output will be used for calculations.
        if describe is True:
            print(frameCopy.describe())


        ''' Of course plot, a picture is a thousand words. Also, streamlit has a problem with
            matplotlib figures being run while streamlit itself is running. That's why the 2nd
            optional parameter exists. '''
        if streamLitRequest is False:
            frameCopy.boxplot(column=[columnName])
            plt.grid(False)
            plt.show()


        ''' Important values to be calculated.
            q1 - # between smallest value and median.
            q3 - # between median and highest value.
            iqr - interquartile range. 25th to 75th percentile.

            First extract the outliers from the specified column. Also, when saving
            the index of the outliers it's possible to get rid of them.

            To begin, calculate the quantiles and iqr. The quantiles are the trickiest
            to deal with. Specifying the values are basically thresholds, meaning changing
            the values will change the results of the overall process this function tries
            to do. '''
        q1 = frameCopy[columnName].quantile(0.25)
        q3 = frameCopy[columnName].quantile(0.75)
        iqr = q3 - q1

        ''' Calculate lower and upper boundaries. A normal formula will look like this:
            "lowerBound = q1 - 1.5 * iqr". Altering the 1.5 to 0.5 "stretched out" the 
            ability to remove outliers. '''
        lowerBound = q1 - 1.35 * iqr
        upperBound = q3 + 1.35 * iqr

        ''' Use list to store all indexes of outliers. It works on 2 simple math conditions.
            1) (self.m_df[columnName] < lowerBound) - Any # smaller than the lower bound.
            2) (self.m_df[columnName] > upperBound) - Any # greater than the upper bound.
            
            Doing both ensures us that outliers on opposite sides are found. '''
        outlierIndexes = frameCopy.index[(frameCopy[columnName] < lowerBound) | (frameCopy[columnName] > upperBound)]

        # Now remove the outliers from a dataframe copy.
        outlierIndexes = sorted(set(outlierIndexes))
        
        frameCopy = frameCopy.drop(outlierIndexes)

        if streamLitRequest is False:
            frameCopy.boxplot(column=[columnName])
            plt.grid(False)
            plt.show()

        return frameCopy




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
