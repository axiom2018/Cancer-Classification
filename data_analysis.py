from os import name
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
from data import Data

''' 

            Data Analysis

This class is for showing information about the dataset being used and possibly visualizations as well.

'''

class DataAnalysis(Data):
    ''' Default arguments will show certain things based on provided arguments during class object creation. 
        This allows the user to see what they would like and not even have to manually call the function
        themselves. '''
    def __init__(self, df, showDataDetails=False, showCountPlot=False, showBoxPlot=False):
        self.m_df = df
        
        if showDataDetails is True:
            self.ShowDataFrameDetails()

        if showCountPlot is True:
            self.CountPlot()

        if showBoxPlot is True:
            self.BoxPlot()



    ''' Streamlit function (constructor). The normal code that displays in the
        terminal uses the other one to make use of variables. However with streamlit,
        variables persist with session_state. ''' 
    def __init__(self):
        pass



    # Streamlit function Display is overriding base class function in Data.py
    def Display(self):
        # Get the dataframe through session state to carry out rest of function code.
        df = st.session_state.df

        st.write('### [Data Analysis] Shows details on the dataframe, visualizations as well.')

        st.write('#### Dataframe: ')
        st.write(df.head(8))

        st.write("")
        st.write("")
        st.write('#### Smaller details:')
        st.write(f'Dataframe shape (Rows/Columns): {df.shape}')
        st.write(f'Unique classes: {df.diagnosis.unique()}')
        st.write('Heatmap:')

        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(df.iloc[:, 1:9].corr(), annot=True)
        st.pyplot(fig)
        



    def ShowDataFrameDetails(self):
        # .head gives first n (default 5) number of rows, when printed.
        print(f'---Head---:\n{self.m_df.head()}\n')

        # .shape gives # of rows and columns in the dataframe.
        print(f'---Shape---: {self.m_df.shape}\n')

        # View all column names dataset has.
        print(f'---Column names---: {self.m_df.columns}\n')

        ''' See the types of data, but more importantly see any missing values from 
            the total amount of rows in every column. Very important to see because
            if any do occur, things must be done. '''
        print(f'---Info---:\n{self.m_df.info()}')



    ''' Even better than the box plot, the violin plot helps visualize the distribution 
        of data which is valuable when you know or suspect the dataset can be imbalanced.
        The violin still has the 1st quartile, median, 2nd quartile, etc just as the box
        plot.
        
        A box plot will show density in the major area, but no so much elsewhere. Also
        a scatter plot will be overloaded with points if the dataset is massive.

        We'll HAVE to apply some a feature scaling technique called standardization to
        GET the proper violin plot. Standardization are when values are completely centered
        around the mean and go along with the new UNIT standard deviation, where unit means
        converting the data features from original units to units of standard deviation.

        The formula is actually simple: 
        X = X - u 
            / 
            o
        u - mean of the feature values
        o - standard deviation of the feature values. '''
    def ViolinPlot(self, plotCertainAmountOfFeatures=False, amountOfFeaturesToPlot=0):
        # Some columns aren't needed for this purpose so get rid of them.
        editedFeatures = self.m_df.drop(['id', 'diagnosis'], axis=1)

        # The previously mentioned formula at work, changes the values.
        resultOfEditedFeatures = (editedFeatures - editedFeatures.mean()) / (editedFeatures.std())

        ''' New dataset being created by concatenating other data together. 0:10 are the range of 
            features to show. '''
        newDataSet = pd.concat([self.m_df.diagnosis, resultOfEditedFeatures.iloc[: , 0:10]], axis=1)

        # Melt changes/unpivots dataframe into long format. Some columns are identifiers, others, measurable.
        newDataSet = pd.melt(newDataSet, id_vars='diagnosis', var_name='features', value_name='value')
        
        ''' Now onto the plotting. Remember this will show a distribution of the classes which is very
            useful when dealing with an imbalanced dataset. 
            
            x, y, hue - Names of the variables in the dataset to be concerned with and display.

            data - The dataset which the violinplot will work from.

            Split - Draws half the violins side by side. 

            inner - The inner representation, the argument defaults to box so if nothing else is specified,
                the representation of the points in the violin interior will be taken care of by a box.
                Instead quart is given for quartile, which gives the quartiles of distribution.
            
            palette - Just colors to make the plot look nice.
        '''
        plt.figure(figsize=(11, 7))
        sns.violinplot(x='value', y='features', hue='diagnosis', data=newDataSet, split=True,
            inner='quart', palette='Set1')
        plt.show()



    ''' Probability density function. 
        
        The concept is simple. It is a likelihood that a variable
        will be a specific value. So values with higher pdf are more likely to happen.

        2 features will be used (radius_mean & area_mean) but that doesn't mean they're the only ones that can
        be used. radius_mean & area_mean were selected due to high correlation as shown in the heat map in
        feature_engineering.py. '''
    def CalculatePDF(self):
        # x values are needed for the .fit function.
        x = self.m_df['area_mean'].to_list()

        # The returning variables are location and scale in the pdf function, so they're necessary.
        mean, std = stats.norm.fit(x)

        ''' Linespace just gets a certain # of values between a range of the first 2 arguments. Here 
            the min and max of the dataframe values that were turned into a list are used.  '''
        xAxis = np.linspace(min(x), max(x), 100)

        # Now of course the probability density function to see values that have a higher likelihood.
        pdf = stats.norm.pdf(xAxis, loc=mean, scale=std)
        
        plt.plot(xAxis, pdf, color='black')
        plt.xlabel('x')
        plt.ylabel('pdf')
        plt.show()



    ''' Cummulative distribution function.
    
        This function is the probability that variable x takes values lower or equal to a 
        specified value. The max has to range between 0 or 1. If the resulting value is
        below or over those values, check for errors in calculations. '''
    def CalculateCDF(self):
        # x values are needed for the .fit function.
        x = self.m_df['area_mean'].to_list()

        # The returning variables are location and scale in the pdf function, so they're necessary.
        mean, std = stats.norm.fit(x)

        ''' Linespace just gets a certain # of values between a range of the first 2 arguments. Here 
            the min and max of the dataframe values that were turned into a list are used.  '''
        xAxis = np.linspace(min(x), max(x), 100)

        cdf = stats.norm.cdf(xAxis, loc=mean, scale=std)

        plt.plot(xAxis, cdf, color='black')
        plt.xlabel('x')
        plt.ylabel('cdf is probability n less than/equal to N')
        plt.show()


    ''' Correlation is for finding out a relationship (be it positive, negative, or neutral)
        between 2 variables. This is definitely used in the world of business to see if one
        thing influences another in any manner at all. A quick example might be if a company
        has used a popular mascot in commercials, and they see a sharp increase in sales, then
        that MIGHT indicate a positive correlation between the 2.
        
        Threshold - Depending on if the correlation values are great enough, show the value. 
            MUST be inbetween 0 and 1 as a float value.
            
        seeCurrentColumnRows - The first column/feature in this dataset is radius_mean so
            it's possible, as shown in the code, to get that particular column and of course
            the rows which are radius_mean, texture_mean, etc. This value being true means
            regardless of the threshold just display them, so later the algorithm will
            pick out ones according TO the threshold for real side by side comparison. 
            
        seeCorrelationHeatMap - Self explanatory. '''
    def Correlation(self, threshold=0.5, seeCurrentColumnRows=False, seeCorrelationHeatMap=False):
        # Only use a few columns for the correlation.
        corrMatrix = self.m_df.iloc[:, 1:9].corr()

        # Get column names for comparisons.
        columnNames = corrMatrix.columns.tolist()

        # Loop through each portion of the dataset via column name and filter the info.
        for name in columnNames:
            ''' Get a portion of the matrix that is the column/feature. For example, accessing
                the radius_mean column will look like:

                                  radius_mean
                radius_mean          1.000000
                texture_mean         0.323782
                perimeter_mean       0.997855
                area_mean            0.987357
                smoothness_mean      0.170581
                compactness_mean     0.506124
                concavity_mean       0.676764 
                
                In this case, the name parameter below will be radius_mean, and therefore it will
                be simple to access all the rows to see radius_mean correlates with others. '''
            curDf = corrMatrix[name].to_frame()

            if seeCurrentColumnRows is True:
                print(f'---{name}--- column:\n{curDf}')
                print('\n')

            ''' Now just do what was mentioned in the previous comment, go through each row
                of a particular column/feature and check the correlation value.
                
                The first line in the for loop gets the numerical value of correlation. 

                The second checks the value with the threshold value, if it matches or exceeds 
                the threshold, that's only the first half. The next goal is to make sure the
                row name doesn't match the column name. It's obvious the same row name and column
                name correlate by a value of 1.0 all the time so skip that. Then of course save 
                the sentence of information to the list if these criteria are met. '''

            for nameOfRow in columnNames:
                val = round(curDf.loc[nameOfRow][0], 3)

                if val >= threshold and nameOfRow != name:
                    print(f'{name} correlation with {nameOfRow} = {val}')

            print('\n')


        if seeCorrelationHeatMap is True:
            self.HeatMap()


    def HeatMap(self):
        plt.figure(figsize=(10, 10))
        matrix = sns.heatmap(self.m_df.iloc[:, 1:9].corr(), annot=True)
        plt.show()



    ''' View how many types of the target variables there are in graph using seaborn. 
        Also terminal to view it later after user exists graph. '''
    def CountPlot(self):
        valueCounts = self.m_df['diagnosis'].value_counts()
        print(f'---value_counts of target variable---:\n{valueCounts}')
        sns.countplot(self.m_df['diagnosis'])
        plt.show()



    ''' Box plots are a bit useful. They demonstrate the distribution of data. Using a
        minimum, q1 (quartile), median, q3, and maximum, the box plot has the ability
        to tell us about outliers. If outliers will be significant in the project, this
        will help identify them. Also determines if the data is symmetrical of not as an
        imbalance could be important to see.
        
        Ex: The outliers will be displayed by little diamonds in the boxplot in the
            function. For M, the top or highest outlier will be 2501. Seeing the max
            value in the area_mean feature/column, we can confirm this with the code:
            "print('Max value is ' + str(np.amax(self.m_df['area_mean'].values)))". The
            result matches the afforementioned outlier. '''
    def BoxPlot(self):
        sns.boxplot(x='diagnosis', y='area_mean', data=self.m_df)
        plt.show()