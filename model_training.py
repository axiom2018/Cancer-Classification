import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression


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

        self.m_x_train, self.m_x_test, self.m_y_train, self.m_y_test = train_test_split(self.m_X, 
                    self.m_y, test_size=0.20, random_state=0)




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


            There's a million different combinations to produce the best results. I'll
            put some in the comments that produced a LOWER result then what's currently
            in the code.

                1) solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))
                    cValue = trial.suggest_loguniform("C", 1e-7, 10.0)

                    Result = 0.947
                
        '''

        # Result = 0.949.
        epochs = trial.suggest_int("max_iter", 200, 500)
        solver = trial.suggest_categorical("solver", ("lbfgs", "newton-cg"))
        cValue = trial.suggest_loguniform("C", 1e-7, 10.0)

        ''' After parameters beforehand are taken care of, then create model. 
            Very important to do because each model has different parameters 
            so optimizing for each one is tricky. '''
        lr = LogisticRegression(max_iter=epochs, solver=solver, C=cValue)

        score = cross_val_score(lr, self.m_X, self.m_y, n_jobs=1, cv=3)
        accuracy = score.mean()
        return accuracy

    
    def ApplyLogisticRegression(self):
        ''' Create the study necessary for it to run trials on the specified function. 
            The optimize function will begin the loop to run repeatedly in order to
            get the best model.'''
        study = optuna.create_study(direction='maximize')
        study.optimize(self.LogisticRegression, n_trials=100)