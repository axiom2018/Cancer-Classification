import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from data import Data

''' 

            Feature Engineering

Feature engineering is an extra few steps that can increase the overall model performance.

'''

class FeatureEngineering(Data):
    def __init__(self, df):
        self.m_df = df

        # A part of feature engineering is getting rid of columns/features. The id column is useless.
        self.m_df.drop('id', axis=1, inplace=True)
        st.session_state.updatedDf.drop('id', axis=1, inplace=True)

        ''' While testing the core functionality in the Display function, this
            boolean helps the functions not run repeatedly if this class was
            last in line the 'listOfClasses' found in main.py. Was used primarily
            for testing purposes to guarantee the code this boolean controls
            wouldn't run multiple times. '''
        self.m_stEncodeAndCorrelationDone = False



    ''' Streamlit function (constructor). The normal code that displays in the
        terminal uses the other one to make use of variables. However with streamlit,
        variables persist with session_state. ''' 
    def __init__(self):
        # Add boolean for encode and correlation so they don't repeatedly happen in the display function.
        self.m_stEncodeAndCorrelationDone = False

        # if 'encodedAndCorrelationBool' not in st.session_state:
        #     st.session_state.encodedAndCorrelationBool = False


    # Streamlit function Display is overriding base class function in Data.py
    def Display(self):
        st.write('### [Feature Engineering] Changing features for 1 reason or another can effect model performance.')
        st.write('')
        st.write('')
        st.write('')

        # First remove the id column, it's not necessary at all.
        st.write(f'##### Remove column/feature [{st.session_state.updatedDf.columns[0]}] because it is not necessary.')
        st.write(f'Dataframe CURRENT shape: {st.session_state.updatedDf.shape}. # of columns/features: {len(st.session_state.updatedDf.columns)}')
        
        st.session_state.df.drop('id', axis=1, inplace=True)
        st.session_state.updatedDf.drop('id', axis=1, inplace=True)

        st.write(f'Dataframe NEW shape: {st.session_state.updatedDf.shape}. # of columns/features: {len(st.session_state.updatedDf.columns)}')


        st.write('')
        st.write('')
        st.write('')
        st.write('##### Since this process effects features/columns, here they all are: ')
        st.write(f'{st.session_state.updatedDf.columns.to_list()}')

        st.write('')
        st.write('')
        st.write('##### Remove features that have high VIF (Variance Inflation factor) which is which features are highly correlated with the data.')

        # if st.session_state.encodedAndCorrelationBool is False:
        if self.m_stEncodeAndCorrelationDone is False:
            self.LabelEncoding(False, True)
            self.Correlation(2, False, False, True)
            st.session_state.encodedAndCorrelationBool = True


    

    ''' Percentile scoring is simple so here's an example with a dataset:

                    Points in game      Percent         Percentile rank
        trisha      57                  57%             25%
        john        80                  80%             75%
        carl        71                  71%             50%
        bob         43                  43%             0%
        roxanne     98                  98%             100%

        Roxanne got the highest points so in the percentile RANK, it's said she
        beat 100% of others. On the opposite, Bob got the lowest so we say he beat
        0% of others. Carl is in the middle, because there are 5 examples (including
        Carl) so half the observations are below 71%. That's true, both Trisha
        and Bob are below that. Yet John and Roxanne are well above. So Carl has
        a 50% percentile rank.

        To see how this applies to real world machine learning, what if a dataset
        has an age column/feature? Then you can see the dataframe but a few peoples
        ages might be 189, 213, 423, etc. That makes no sense whatsoever so viewing
        them as outliers is a proper approach. Maybe doing something like
        df['age'].quantile(0.90) to get the value that represents being above
        90% quantile would be appropriate. Then any value greater than that, get rid 
        of it. 
        
        
        column - Be sure a proper column name is provided. User PROBABLY won't remember 
            them so if this function is called with no arguments, force them to pick one 
            and provide it. 
            
        showSteps - There's a lot to see if this function is used. But if they just want
            to get the result without the steps then this should be false.

        removeOutliers - It's also important to say not everything might be a true "removable"
            outlier. If there is a value that's super high, maybe it's valid. So
            removal might end up hurting more. Always be careful when dealing with
            outliers!
            
        high/low - These are the quantile decimal values. For example, if high is 90, then
            in the .quantile function used on the given column will return a particular
            float value that basically is the equivalent of saying "Okay anything above
            this particular value will be considered as an outlier". 
            
                So with the above said, this is probably the most important part of this
                long comment. There is NOT "for sure" or "guaranteed" values to remove
                all outliers from any feature/column. Experimenting with these values
                is key. Which is why default values are given but the user is free 
                to mess around with these. '''
    def OutliersPercentile(self, column=None, showSteps=False, removeOutliers=False, high=0.90, low=0.02):
        if column is None:
            print('-----Please provide a column/feature name in the dataset listed below for this function.-----\n')
            print(f'Column names:\n{self.m_df.columns[1:]}\n')
            return

        ''' With selected column, get the number that represents being better than a 
            certain percentile. In this case, it's being better than the value of the
            argument "high", a decimal value. If high is 0.90, this line will find
            in whatever column, get the values above a 90th percentile. '''
        max = self.m_df[column].quantile(high)
        min = self.m_df[column].quantile(low)

        if showSteps is True:
            print(f'The float equivalent of the MAX {high}th percentile with column {column} is: {max}.')
            print(f'The float equivalent of the MIN {low}th percentile with column {column} is: {min}\n')

        # View all values above the high percentile in the dataframe
        maxColumnDf = self.m_df[self.m_df[column] > max]
        minColumnDf = self.m_df[self.m_df[column] < min]
        if showSteps is True:
            print(f'Values in dataframe that surpass the quantile {high} are:\n{maxColumnDf[column]}\n')
            print(f'Values in dataframe that are below the quantile {low} are:\n{minColumnDf[column]}')

        
        # Viewing plots helps get a real handle on things so why not?
        sns.boxplot(x=column, data=self.m_df)
        plt.show()

        # Check if user wanted to remove outliers from actual dataframe.
        if removeOutliers is True:
            originalShape = self.m_df.shape

            # Checking for points less than max but greater than min, removing outliers in both directions.
            self.m_df = self.m_df[(self.m_df[column] < max) & (self.m_df[column] > min)]

            print(f'[Percentile-Outlier Removal] Original df shape: {originalShape}. Post outlier removal shape: {self.m_df.shape}')               
            sns.boxplot(x=column, data=self.m_df)
            plt.show()

        # If not, just create a new dataframe so the original isn't altered.
        else:
            dummyDf = self.m_df[(self.m_df[column] < max) & (self.m_df[column] > min)]

            print(f'[Percentile-No outlier removal] Original df shape: {self.m_df.shape}. Shape with no outliers: {dummyDf.shape}')
            sns.boxplot(x=column, data=dummyDf)
            plt.show()

        return self.m_df



    # Standard deviations can be used in a "range" manner to eliminate outliers in a specific range.
    def OutliersStandardDeviation(self, column=None, showSteps=False, removeOutliers=False, range=2):
        if column is None:
            print('-----Please provide a column/feature name in the dataset listed below for this function.-----\n')
            print(f'Column names:\n{self.m_df.columns[1:]}\n')
            return
        
        ''' A histogram is necessary because it VISUALLY helps to explain the standard 
            deviation (sd) approach. Sd's can be used in ranges. For example, if the mean
            of a given dataset column/feature is 70, and a given datapoint is 93, the point
            is trying to figure out how many sd RANGES the datapoint is from the mean. If
            a dataset is normally distributed then most datapoints should be in 1 sd range.
            
            Then the case with outliers gets a bit clearer. The further sd ranges a datapoint
            is, it can be an outlier. Normally 3 sd range is used, but playing around with 
            values is good for educational purposes. '''
        if showSteps is True:
            plt.hist(self.m_df[column], bins=20, rwidth=0.8)
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.show()

        ''' This is the simple formula. Remember that in order to find how
            far the datapoint is from the mean, well the mean is necessary.
            
            Then we + or - it from the range. 
                - Means the minimum or the lower side.
                + Means the maximum or the higher side.
                
            Of course "sd" means Standard Deviation so that's definitely 
            critical to the formula. The given values means "any value greater
            than n, will be an outlier" or "any value smaller than n, will
            be an outlier". Those are both cases for max and min.  '''
        max = self.m_df[column].mean() + (range * self.m_df[column].std())
        min = self.m_df[column].mean() - (range * self.m_df[column].std())

        # Show the values calculated
        if showSteps is True:
            print(f'Values smaller than this: {min} will be viewed as outliers.')
            print(f'Values greater than this: {max} will be viewed as outliers.\n')

        # Also show the outliers themselves.
        newDf = self.m_df[(self.m_df[column] < max) & (self.m_df[column] > min)]
        if showSteps is True:
            print(f'Observations in dataframe that are viewed as outliers:\n{newDf[column]}\n')
            print(f'Old df shape: {self.m_df.shape}. New df shape: {newDf.shape}')
            print(f'{self.m_df.shape[0] - newDf.shape[0]} outliers affected.')


        # Check if user wanted to remove outliers from actual dataframe.
        if removeOutliers is True:
            originalShape = self.m_df.shape
            
            # Checking for points less than max but greater than min, removing outliers in both directions.
            self.m_df = self.m_df[(self.m_df[column] < max) & (self.m_df[column] > min)]
            
            print(f'[Standard deviation-Outlier Removal] Original df shape: {originalShape}. Post outlier removal shape: {self.m_df.shape}')
            plt.hist(self.m_df[column], bins=20, rwidth=0.8)

        # If not, just create a new dataframe so the original isn't altered.
        else:
            dummyDf = self.m_df[(self.m_df[column] < max) & (self.m_df[column] > min)]

            print(f'[Standard deviation-Outlier Removal] Original df shape: {self.m_df.shape}. Shape with no outliers: {dummyDf.shape}')
            plt.hist(dummyDf[column], bins=20, rwidth=0.8)
        
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

        return self.m_df



    ''' Z scores will give numbers that tells us how many standard deviation (sd) ranges 
        a data point is away from the mean. So it's a cousin of the standard deviation 
        attempt above, still pretty cool though. For example, if a datapoint is 3 sd's 
        away, the z score will be 3. The formula is:
        
        z = x - u
            /
            o
        
        x - The datapoint value. Say if the mean is 50 and there can be a datapoint of 78.
        u - The mean value.
        o - Standard deviation. 
        
        This function will assist only SHOW the z score to remove certain data points.  '''
    def OutliersZScore(self, column=None, zscore=2):
        if column is None:
            print('-----Please provide a column/feature name in the dataset listed below for this function.-----\n')
            print(f'Column names:\n{self.m_df.columns[1:]}\n')
            return

        # Get a copy of the main dataframe.
        df = self.m_df.copy()

        print(f'Calculating z score with column {column}.')

        # Use the formula and print an edited dataframe with only the argument column and z score.
        df['Z score'] = (df[column] - df[column].mean()) / df[column].std()
        df = df[[column, 'Z score']]
        print(df)

        # Show the outlier results.
        print(f'---Outliers greater than z score {zscore}:')
        print(df[df['Z score'] > zscore])

        print(f'\n\n---Outliers less than z score {zscore}:')
        print(df[df['Z score'] < -zscore])





    # Label encoding is solely for categorical variables of course.
    def LabelEncoding(self, showSteps=False, streamLitRequest=False):
        le = LabelEncoder()

        if showSteps is True:
            self.m_df['diagnosis'] = le.fit_transform(self.m_df['diagnosis'])

            print('Result of label encoding for categorical column:')
            print(self.m_df['diagnosis'])

            print('\nUnique values of categorical column:')
            print(self.m_df['diagnosis'].value_counts())
        

        ''' This function, if used with a stream lit request, must edit the
            categorical feature/column in the updated dataframe. It's not 
            reasonable to use anything but the updated dataframe which is
            stored in the streamlit session state because that dataframe could
            of been altered in the last class. '''
        if streamLitRequest is True:
            st.session_state.updatedDf['diagnosis'] = le.fit_transform(st.session_state.updatedDf['diagnosis'])
            st.write('')
            st.write('')
            st.write('')
            st.write(f'##### First encode the {st.session_state.updatedDf.columns[0]} feature/column:')
            st.write(st.session_state.updatedDf['diagnosis'])
            st.write('')
            st.write('')
            st.write('')



    ''' This function helps the Correlation function. It will calculate the
        Vif scores for each column/feature in a dataframe and return the
        results.

        Simply create new columns in the new dataframe and in the vif_score
        feature/column, use list comprehension to GET the proper vif score.
        Also sort it to make it easier to see where the highest vif values are. '''
    def CalculateVifScore(self, df):
        vif = pd.DataFrame()
        vif.astype(float)
        vif['Features'] = df.columns
        vif['Vif_Score'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
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
    def Correlation(self, threshold=2, showHeatMap=False, showSteps=False, streamLitRequest=False):
        if showSteps is True and streamLitRequest is False:
            plt.figure(figsize=(10, 10))
            matrix = sns.heatmap(self.m_df.iloc[:, 1:9].corr(), annot=True)
            plt.show()

        # Create the table of vif values and print it.
        vdf = None 
        
        # Pass proper dataframe just incase the request to do so is streamlit based.
        if streamLitRequest is False:
            vdf = self.CalculateVifScore(self.m_df)
        else:
            vdf = self.CalculateVifScore(st.session_state.updatedDf)
            st.write('')
            st.write('###### Vif Scores (Higher the score, the more correlated a feature is with rest of data):')
            st.write(vdf)

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


        # Give suspicious features to streamlit screen.
        if streamLitRequest is True:
            st.write(f'Top features to investigate: {featuresToLookInto}')
            st.write('')
            st.write('')
            st.write('')
            st.write('Before getting the new table/VIF scores. Steps must be taken.')
            st.write('1) Finding the highest values')
            st.write('2) Dropping the highly correlated features/column from the dataframe.')
            st.write('3) Calculate Vif scores again.')
            st.write(f'4) Do steps 1-3 a certain # of times. This approach will have it done {threshold} times.')
            st.write('')


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
            col1 = None
            col2 = None
            if streamLitRequest is False:
                col1 = self.m_df['diagnosis']
                col2 = self.m_df[column]
            else:
                col1 = st.session_state.updatedDf['diagnosis']
                col2 = st.session_state.updatedDf[column]
            
            # col1 = self.m_df['diagnosis']
            # col2 = self.m_df[column]
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

            if streamLitRequest is False:
                # Drop the column and get the vif ratings again.
                self.m_df.drop([featuresToLookInto[i]], axis=1, inplace=True)
            else:
                st.session_state.updatedDf.drop([featuresToLookInto[i]], axis=1, inplace=True)


            # Create the new table of vif values and print it.
            if streamLitRequest is False:
                vdf = self.CalculateVifScore(self.m_df)
            else:
                vdf = self.CalculateVifScore(st.session_state.updatedDf)

            i -= 1

        if showSteps is True:
            print(f'-----Final VIF results table:-----\n{vdf}\n')
            print(f'-----Current dataframe:-----\n{self.m_df}')

        if streamLitRequest is True:
            st.write('')
            st.write('')
            st.write('')
            st.write(f'##### After removing the previous features {featuresToLookInto} the final VIF dataframe looks like:')
            st.write(vdf)

            st.write('')
            st.write('')
            st.write('')
            st.write(f'##### updatedDf feature/column length is {len(st.session_state.updatedDf.columns)} & the dataframe itself looks like: ')
            st.write(st.session_state.updatedDf.head(5))