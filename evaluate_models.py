from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

''' 

            Model Training

View accuracy, confusion matrices, and those sorts of things.

'''

class EvaluateModels:
    def __init__(self, models, xTest, yTest):
        self.m_models = models
        self.m_xTest = xTest
        self.m_yTest = yTest
        self.m_currentBestModel = None
        self.m_currentBestModelScore = 0


    ''' Confusion Matrix
    
        It gives a 2 by 2 matrix of cancer (malignant or m) and not cancer (benign or b).

        The top row shows how many that were cancerous in total, left and right.

        Top left - Prediction is cancer, ground truth was cancer. 
        Top right - Prediction is NOT cancer, ground truth was cancer.

        Bottom left - Prediction is cancer, ground truth was not cancer.
        Bottom right - Prediction is not cancer, ground truth was not cancer. 
        
        Use every model and get the predictions for the model on the xtest values. 
        Then compare with the first argument given in the confusion_matrix, which is
        ground truths, or the CORRECT values. '''
    def ViewConfusionMatrices(self):
        for model in self.m_models:
            cm = confusion_matrix(self.m_yTest, model[1].predict(self.m_xTest))
            print(f'Confusion matrix for {model[0]} is:\n{cm}')


    def ViewClassificationReport(self):
        for model in self.m_models:
            cr = classification_report(self.m_yTest, model[1].predict(self.m_xTest))
            print(f'{model[0]} classification report:\n{cr}')


    # This function will pick out a certain model based on its overall accuracy. 
    def ViewAccuracy(self):
        for model in self.m_models:
            accScore = round(accuracy_score(self.m_yTest, model[1].predict(self.m_xTest)), 2)
            print(f'{model[0]} accuracy score is {accScore}')

            ''' If there is no current best model, set the first one. Afterwards when
                the current best model value is NOT none, comparisons must begin with
                the current best and whatever model is currently displaying its score
                above.
                
                Simply compare the score of the model inside the 'model' variable
                and see if it's greater than the self.m_currentBestModel score. If so,
                that's a new best model! '''
            if self.m_currentBestModel is None:
                self.m_currentBestModel = model
                self.m_currentBestModelScore = accScore
            else:
                if accScore > self.m_currentBestModelScore:
                    print(f'''\n{model[0]} is new best model with a score of {accScore} which beat previous score of {self.m_currentBestModelScore}
                    set by {self.m_currentBestModel[0]}\n''')

                    self.m_currentBestModel = model
                    self.m_currentBestModelScore = accScore