from numpy.core.fromnumeric import var
import optuna
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


''' 

            Model Training

Splitting the dataframe up and building the model will be this classes responsibility. 


    All models will be used with Optune since it can tune hyperparameters. That's key 
    to having good models because manually playing with them isn't ideal as it's time consuming.

'''


class ModelTraining:
    def __init__(self, df):
        self.m_df = df

        ''' X - The features that can detect if a person has cancer or not.
            y - The target/real values that confirm if a person has cancer or not. '''
        self.m_X = self.m_df.iloc[:, 1:].values
        self.m_y = self.m_df.iloc[:, 0].values

        # Simple list to hold all models. For use in TrainModels function.
        self.m_models = []
        
        # Out of all models tested, this will be the main one used.
        self.m_bestModel = None

        self.m_x_train, self.m_x_test, self.m_y_train, self.m_y_test = train_test_split(self.m_X, 
                    self.m_y, test_size=0.20, random_state=0)

        ''' No need for the flood of output Optuna produces in the terminal. 
            Turn it back on with "optuna.logging.enable_default_handler()" '''
        optuna.logging.disable_default_handler()


    
    def LogisticRegression(self, trial):
        ''' Search space, basically allows customization on model arguments. 

            epochs - Epochs or iterations is the value needed for convergence
                to happen among solvers. In general can help the model grow e
                ven by a little bit. Think of an athlete that runs 30 laps instead of 20. 
                It's called max_iter in the LogisticRegression model itself.
            
            solver - The algorithm to use for optimization. There are several but
                ones that WON'T be used are sag & saga as those are faster for larger
                datasets. Of course the cancer dataset isn't too big so no need for them.
 

            C - Regularization, which can make feature weights drop close to 0 or 
                absolute 0. DEFINITELY a game changer.


            There's a million different combinations to produce the best results. As seen
            with the search space arguments.

            make_pipeline was put to use to rid the converge warning that starts off as 
            "Increase the number of iterations (max_iter) or scale the data as shown in:". 
            One recommended solution was to make use of a pipeline and here it is.
                
        '''
        C = trial.suggest_loguniform("C", 1e-7, 10.0) 
        solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))
        max_iter = trial.suggest_int("max_iter", 4000, 4000)

        pipe = make_pipeline(preprocessing.StandardScaler(), 
                LogisticRegression(C=C, solver=solver, max_iter=max_iter))

        pipe.fit(self.m_x_train, self.m_y_train)
        
        return pipe.score(self.m_x_test, self.m_y_test)



    def RandomForest(self, trial):
        ''' Defining the search space. 
        
            criterion - How good are the splits? Either gini or entropy determines that.
                Since there's only 2 options for criterion here, might as well choose both.

            max_depth - How deep the trees will be.

            n_estimators - # of trees in the forest. Arguably the most vital parameter.

            Current best parameters are currently in the function:

                100 trials

                Best is trial 15 with value: 0.9701383087347999.
                parameters: {'criterion': 'entropy', 'max_depth': 17, 'n_estimators': 80}.
        
        '''
        criterion = trial.suggest_categorical("criterion", ("gini", "entropy"))
        maxDepth = trial.suggest_int("max_depth", 1, 20)
        estimators = trial.suggest_int("n_estimators", 1, 100)

        # Initialize the model, then get the score and accuracy.
        rf = RandomForestClassifier(n_estimators=estimators, criterion=criterion, max_depth=maxDepth)

        score = cross_val_score(rf, self.m_X, self.m_y, n_jobs=1, cv=3)
        accuracy = score.mean()
        return accuracy


    
    def DecisionTree(self, trial):
        ''' Defining the search space. 
        
            criterion - How good are the splits? Either gini or entropy determines that.
                Since there's only 2 options for criterion here, might as well choose both.

            splitter - For choosing splits. The 2 options are best and random. Selecting
                best in theory should definitely work better.

            max_depth - How deep the trees will be.

            min_samples_split - # of samples needed before node splitting.

            Current best parameters:

                100 trials

                Best is trial 73 with value: 0.9543488350505894.
                parameters: {'criterion': 'entropy', 'splitter': 'random', 'max_depth': 16, 'min_samples_split': 3}
        '''
        criterion = trial.suggest_categorical("criterion", ("gini", "entropy"))
        splitter = trial.suggest_categorical("splitter", ("best", "random"))
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 5)

        # Initialize the model, then get the score and accuracy.
        dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, 
            min_samples_split=min_samples_split)

        score = cross_val_score(dt, self.m_X, self.m_y, n_jobs=1, cv=3)
        accuracy = score.mean()
        return accuracy



    def Svm(self, trial):
        C = trial.suggest_loguniform("C", 0.5, 1.0)
        kernel = trial.suggest_categorical("kernel", {"rbf", "linear", "poly"})

        # Initialize the model, then get the score and accuracy.
        svc = SVC(C=C, kernel=kernel)
        svc.fit(self.m_x_train, self.m_y_train)

        score = cross_val_score(svc, self.m_X, self.m_y, n_jobs=1, cv=3)
        accuracy = score.mean()
        return accuracy



    def NaiveBayes(self, trial):
        var_smoothing = trial.suggest_loguniform("var_smoothing", 1e-2, 10.0)

        nb = GaussianNB(var_smoothing=var_smoothing)
        nb.fit(self.m_x_train, self.m_y_train)
        
        return nb.score(self.m_x_test, self.m_y_test)




    ''' All "apply" functions will be used to create study objects to 
        run the core algorithms themselves. '''
    def ApplyLogisticRegression(self, showSteps, epochs=100):
        if showSteps is True:
            print('Starting logistic regression optimization.')

        ''' Create the study necessary for it to run trials on the specified function. 
            The optimize function will begin the loop to run repeatedly in order to
            get the best model.'''
        study = optuna.create_study(direction='maximize')
        study.optimize(self.LogisticRegression, n_trials=epochs)

        # Brings back the best result from various parameters being tested.
        trial = study.best_trial

        ''' Get the individual arguments used in the real algorithm function, 
            LogisiticRegression. Must make sure they match. Then afterwards
            just initialize the model in here as well, train it, and return it. '''
        solver = trial.params["solver"]
        iterations = trial.params["max_iter"]
        C = trial.params["C"]

        lr = LogisticRegression(solver=solver, max_iter=iterations, C=C)
        lr.fit(self.m_X, self.m_y)

        if showSteps is True:
            print('Logistic regression optimized.')

        return lr

        

    def ApplyRandomForest(self, showSteps, epochs=100):
        if showSteps is True:
            print('Starting random forest optimization.')

        study = optuna.create_study(direction='maximize')
        study.optimize(self.RandomForest, n_trials=epochs)

        trial = study.best_trial

        criterion = trial.params["criterion"]
        max_depth = trial.params["max_depth"]
        estimators = trial.params["n_estimators"]

        rf = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=estimators)
        rf.fit(self.m_X, self.m_y)

        if showSteps is True:
            print('Random forest optimized.')

        return rf



    def ApplyDecisionTree(self, showSteps, epochs=100):
        if showSteps is True:
            print('Starting decision tree optimization.')

        study = optuna.create_study(direction='maximize')
        study.optimize(self.DecisionTree, n_trials=epochs)

        trial = study.best_trial

        criterion = trial.params["criterion"]
        splitter = trial.params["splitter"]
        max_depth = trial.params["max_depth"]
        min_samples_split = trial.params["min_samples_split"]

        dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
            min_samples_split=min_samples_split)
        
        dt.fit(self.m_X, self.m_y)

        if showSteps is True:
            print('Decision tree optimized.')

        return dt



    def ApplySVM(self, showSteps, epochs=100):
        if showSteps is True:
            print('Starting svm optimization.')

        study = optuna.create_study(direction='maximize')
        study.optimize(self.Svm, n_trials=epochs)

        trial = study.best_trial

        C = trial.params["C"]
        kernel = trial.params["kernel"]

        svc = SVC(C=C, kernel=kernel)
        svc.fit(self.m_x_train, self.m_y_train)

        if showSteps is True:
            print('Svm optimized.')

        return svc



    def ApplyNaiveBayes(self, showSteps, epochs=100):
        if showSteps is True:
            print('Starting naive bayes optimization.')

        study = optuna.create_study(direction='maximize')
        study.optimize(self.NaiveBayes, n_trials=epochs)

        trial = study.best_trial

        var_smoothing = trial.params["var_smoothing"]

        nb = GaussianNB(var_smoothing=var_smoothing)
        nb.fit(self.m_x_train, self.m_y_train)

        if showSteps is True:
            print('Naive bayes optimized.')

        return nb



    ''' The main function of this class. In order to get the voting classification
        to fit in, there needs to be multiple different algorithms ready. In order
        to have different ones ready, of course the multiple above functions utilizing
        Optuna must be ran. This function will run them all, get their trained models,
        and then use a voting classification to get the best model. '''
    def TrainModelsVotingClassifier(self, showSteps=False):
        if showSteps is True:
            print('---Model training beginning---')

        self.m_models.append(('Logistic Regression', self.ApplyLogisticRegression(showSteps)))
        self.m_models.append(('Random Forest', self.ApplyRandomForest(showSteps, 50)))
        self.m_models.append(('Decision Tree', self.ApplyDecisionTree(showSteps)))
        self.m_models.append(('SVM', self.ApplySVM(showSteps, 20)))
        self.m_models.append(('Naive Bayes', self.ApplyNaiveBayes(showSteps)))

        if showSteps is True:
            print('---Model training is done.---\n')

        # Initialize voting classifier and fit on x/y train data
        vc = VotingClassifier(estimators=self.m_models, voting='hard')
        
        ''' Quick cross validation used to see how the models are doing.
        
            KFold - Splits data into k amount of folds.
        
            Scores - The function cross_validation will return a dictionary and the main
                key of interest is "test_score" to of course means the scores of every
                model that was used. Also apply the KFold #. 
                
            Then print out the test scores. '''
        kf = KFold(shuffle=True, random_state=11)
        scores = cross_validate(vc, X=self.m_x_train, y=self.m_y_train, cv=kf)
        testScores = scores['test_score']
        print(f'Cross validation scores:\n{testScores}')

        # Fit then get predictions and finally see the accuracy.
        vc.fit(self.m_x_train, self.m_y_train)
        vcPred = vc.predict(self.m_x_test)

        print(f'Accuracy score of voting classifier:\n{accuracy_score(self.m_y_test, vcPred)}\n')