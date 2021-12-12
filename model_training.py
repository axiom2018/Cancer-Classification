from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import xgboost as xgb


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

        # Save the train/test sets here.
        self.m_xTrain = None
        self.m_xTest = None
        self.m_yTrain = None
        self.m_yTest = None

        # Get models ready.
        self.m_models = []

        self.m_models.append(('Logistic Regression', LogisticRegression(random_state=0, max_iter=200))) # Increase iters for convergence in Adaboost function.
        self.m_models.append(('Decision Tree', DecisionTreeClassifier(criterion='entropy', random_state=0)))
        self.m_models.append(('Random Forest', RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)))


    ''' There is still a bit of feature engineering to do which is scaling, and that
    does require the a list that train_test_split returns. '''
    def GetTrainAndTestSets(self):
        x_train, x_test, y_train, y_test = train_test_split(self.m_X, self.m_y, test_size=0.20, random_state=0)
        self.m_xTrain = x_train
        self.m_xTest = x_test
        self.m_yTrain = y_train
        self.m_yTest = y_test
        return x_train, x_test, y_train, y_test

    
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
        print('---Beginning model training.---')

        for model in self.m_models:
            model[1].fit(xtrain, ytrain)

            if printTrainingAccuracy is True:
                print(f'{model[0]} training accuracy: {model[1].score(xtrain, ytrain)}')
        
        print('---Ending Model training.---\n')


    def XgBoostModel(self):
        print('---Begin Xgboost model training.---')
        xgbTrain = xgb.DMatrix(self.m_xTrain, label=self.m_yTrain)
        xgbTest = xgb.DMatrix(self.m_xTest, label=self.m_yTest)

        ''' Picking model parameters is key. To be honest there are SO many of them 
            here are the ones chosen. Arguably the hardest part.
            
            1) max_depth - Maximum depth of a tree. Default is 6 but increasing this
                will probably make the model overfit and that's not desirable considering
                the entire POINT of boosting is to hopefully INCREASE performance. Also,
                the deeper the tree, the more memory is consumed.
            
            2) eta - When boosting step is done, get the new feature weights and shrink 
                them.
            
            3) objective - Softmax, this will be the most likely classifcation for each
                sample.
            
            4) num_class - Only 2, because there are only 2 classes to choose from of course.
                Either M or B.
            
            And of course epochs just means iterations.
            
            Then just train the model, get predictions and view the results. '''
        param = {
            'max_depth': 4,
            'eta': 0.3,
            'objective':'multi:softmax',
            'num_class': 2}

        epochs = 10

        xgbModel = xgb.train(param, xgbTrain, epochs)
        predictions = xgbModel.predict(xgbTest)

        print(f'Xgboost testing accuracy: {accuracy_score(self.m_yTest, predictions)}')
        print('---Ending Xgboost model training.---\n')


    def GetXgbModel(self):
        return self.m_xgbModel


    ''' Adaboost is a model worth considering. The ada tree (at) differs in some ways 
        to a random forest (rf).
        
        1)  rf - Makes large trees, no real outright max depth.
            at - Has stumps, which are just a top node and 2 leaves. This are also called
                weak learners. Not great at classifications because a stump can only
                look at literally ONE variable.
        
        2)  rf - All trees have an equal vote for classification.
            at - Larger stumps get more credit in the final classification than smaller stumps.

        3)  rf - Order doesn't matter with the trees made in an rf
            at - Order DOES matter. It's like a chain, the errors the first stump makes, will
                help the 2nd stump. Etc, etc.
        

        In the constructor I stated that the logistic regression model has its iterations increased
            to solve the convergence issue that previously went on in the function. That issue has
            definitely been fixed.
    '''
    def AdaBoostModel(self):
        print('---Begin Adaboost model training.---')
        # Get the log reg model.
        logModel = self.m_models[0][1]

        # Get the adaboost object.
        adaClass = AdaBoostClassifier(n_estimators=50, base_estimator=logModel, learning_rate=1)

        # Train and get the score.
        adaModel = adaClass.fit(self.m_xTrain, self.m_yTrain)
        yPred = adaModel.predict(self.m_xTest)

        print(f'Adaboost training accuracy: {adaModel.score(self.m_xTrain, self.m_yTrain)}')
        print(f'Adaboost testing accuracy: {accuracy_score(self.m_yTest, yPred)}')
        print('---Ending Adaboost model training.---\n')

        # Turn the model into a tuple with its name and then add it to the list of models
        self.m_models.append(('Adaboost', adaModel))

    
    ''' Gradient boost model. Very simple to use and set up, especially with the classifier. Just
        like Xgboost model, choosing the correct argument values. 
        
        After some studying, the most common ones to use are:
        
        1) Criterion - Loss function to get the proper threshold.
        
        2) learning_rate - Alters the contribution of each tree.
        
        3) max_depth - The max depth of each tree.
        
        4) n_estimators - # of trees to MAKE. Default 100. 
        
         '''
    def GradientBoost(self):
        gb = GradientBoostingClassifier()
        gb.fit(self.m_xTrain, self.m_yTrain)
        self.m_models.append(('Gradient Boost', gb))


    # For evaluate models class.
    def GetModels(self):
        return self.m_models