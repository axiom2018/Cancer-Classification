from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

''' 

            Feature Engineering

Feature engineering is an extra few steps that can increase the overall model performance.

'''

class FeatureEngineering:
    ''' Regarding the default argument here, it's the same concept as the default arguments
        in data_analysis.py. If user provides true, the heat map will show. Allowing the
        user to see the heat map without calling the function on the class object manually. '''
    def __init__(self, df, showHeatMap=False):
        self.m_df = df

        if showHeatMap is True:
            self.ShowHeatMap()


    ''' 1) Encoding
    
        Part of feature engineering is encoding since models can't work with actual text
        but instead use numbers. Using LabelEncoder with the breast cancer dataset will
        work fine, especially since the categorical column called 'diagnosis' has only 
        2 unique types which are B or M. b=0, m=1 '''
    def EncodeCategoricalColumns(self):
        self.m_labelEncoder = LabelEncoder()
        self.m_df['diagnosis'] = self.m_labelEncoder.fit_transform(self.m_df['diagnosis'])



    ''' 2) Heat map
    
        The heat map is a good way to show correlation in the dataset. It won't show all the 
        features/columns because there's too many to show in a decent fashion. Then take notice 
        to the 'diagnosis' label and how it correlates to every other feature in the heatmap. 
        Because the goal here is to get as many highly correlated features to 'diagnosis', since 
        'diagnosis' IS the target value of course. '''
    def ShowHeatMap(self):
        plt.figure(figsize=(10, 10))
        matrix = sns.heatmap(self.m_df.iloc[:, 1:12].corr(), annot=True)
        plt.show()


    ''' 3) Correlation
    
        To see the default correlations in the dataset, just uncomment the very first print
        statement in the function. Use what's printed to compare to the dataframe after
        this function is complete

        The dataset has plenty of features. It's important to see how many features correlate
        well with the target value which is 'diagnosis'. There can be a certain threshold,
        which is a number, that can be applied to each pair (that is 'diagnosis' and another
        independent variable) to decide whether to use that feature or not. Whether it correlates
        well or not. Must keep in mind that all categorical columns MUST be encoded before hand
        for those once categorical columns can have any correlation values when using .corr() 
        function. '''
    def UpdateFeaturesForCorrelation(self):
        ''' Get the columns with a slice. The first column name is id, which is useless. Then 
            the second is diagnosis, which is the column to be used against every OTHER relevant 
            column to check correlations.
            
            Then use a threshold float value to compare with the corrValue to see how well
            a feature/column correlates with 'diagnosis'. Tuning this value means picking more
            or less features. Then a list is defined to get all the columns that will be
            removed or dropped from the dataset if they fall below the threshold. 
            
            The for loop will begin showing the correlations already described above. Append 
            column names to the list. '''

        print('---Begin updating dataset for more correlated features.---')
        values = self.m_df.columns.tolist()
        values = values[2:]

        threshold = 0.6
        columnsToRemove = []

        for column in values:
            corrValue = self.m_df['diagnosis'].corr(self.m_df[column])
            # print(f'{column} is correlated with diagnosis with value {corrValue}.') # Uncomment to show all correlation values with diagnosis.

            if corrValue < threshold:
                columnsToRemove.append(column)

        for column in columnsToRemove:
            self.m_df.drop(column, inplace=True, axis=1)

        # Also drop the id column, it doesn't have any use.
        self.m_df.drop('id', inplace=True, axis=1)
        print('---Updating dataset for more correlated features done.---')
    

    ''' 4) Scaling
    
        Helps adjust the magntitude of the features, which are getting the values in the 
        columns into a specific range such as 0-100, or could be 0-1. '''
    def UseScaling(self, xTrain, xTest):
        sc = StandardScaler()
        return sc.fit_transform(xTrain), sc.fit_transform(xTest)