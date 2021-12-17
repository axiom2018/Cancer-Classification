import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

''' 

            Data Analysis

This class is for showing information about the dataset being used and possibly visualizations as well.

'''

class DataAnalysis:
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


    def ShowDataFrameDetails(self):
        # .head gives first n (default 5) number of rows, when printed.
        print(f'---Head---:\n{self.m_df.head()}\n')

        # .shape gives # of rows and columns in the dataframe.
        print(f'---Shape---: {self.m_df.shape}\n')

        # View all column names dataset has.
        print(f'---Column names---: {self.m_df.columns}\n')

        # See what TYPE of variable each column holds. Very useful to see what approach might be needed for manipulation of the data.
        print(f'---Types of data in columns---:\n{self.m_df.dtypes}\n')

        # Find empty values if any exist. If so, using dropna may be necessary. 
        print(f'---Are there missing entries in the data?---:\n{self.m_df.isna().any()}')


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