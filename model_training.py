from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

''' 

            Model Training

Splitting the dataframe up and building the model will be this classes responsibility.

'''

class ModelTraining:
    def __init__(self, df):
        self.m_df = df

        ''' X will be the features that can detect if the patient has cancer or not.
            y will be the real values that indicate if that patient has cancer or not. '''
        self.m_X = self.m_df.iloc[:, 1:].values
        self.m_y = self.m_df.iloc[:, 0].values

        ''' Get models ready.'''
        self.m_models = []

        self.m_models.append(('Logistic Regression', LogisticRegression(random_state=0)))
        self.m_models.append(('Decision Tree', DecisionTreeClassifier(criterion='entropy', random_state=0)))
        self.m_models.append(('Random Forest', RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)))


    ''' There is still a bit of feature engineering to do which is scaling, and that
    does require the a list that train_test_split returns. '''
    def GetTrainAndTestSets(self):
        return train_test_split(self.m_X, self.m_y, test_size=0.20, random_state=0)

    
    ''' Of course, need to use some models on the data.
    
        1) LogisticRegression - Simple model,perfect for this case because it uses
            binary classification.

        2) DecisionTreeClassifier - Decision trees seem to be logical to try since they
            make choices to traverse down a tree. Sometimes values can be compared
            to see where to traverse next.
        
        3) RandomForestClassifer - Literally a forest of decision trees.

        Then print some accuracys that occurred while they were training IF the default
        argument is true. '''
    def Train(self, xtrain, ytrain, printTrainingAccuracy=False):
        print('\n---Beginning model training.---\n')
        for model in self.m_models:
            model[1].fit(xtrain, ytrain)

            if printTrainingAccuracy is True:
                print(f'{model[0]} training accuracy: {model[1].score(xtrain, ytrain)}\n\n')
        print('\n---Model training done.---\n')

    # For evaluate models class.
    def GetModels(self):
        return self.m_models