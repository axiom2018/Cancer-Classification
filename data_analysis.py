import seaborn as sns
import matplotlib.pyplot as plt

''' 

            Data Analysis

This class is for showing information about the dataset being used and possibly visualizations as well.

'''

class DataAnalysis:
    ''' Default arguments will show certain things based on provided arguments during class object creation. 
        This allows the user to see what they would like and not even have to manually call the function
        themselves. '''
    def __init__(self, df, showDataDetails=False, showCountPlot=False):
        self.m_df = df

        if showDataDetails is True:
            self.ShowDataFrameDetails()

        if showCountPlot is True:
            self.ShowCountPlot()


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
    def ShowCountPlot(self):
        valueCounts = self.m_df['diagnosis'].value_counts()
        print(f'---value_counts of target variable---:\n{valueCounts}')
        sns.countplot(self.m_df['diagnosis'])
        plt.show()