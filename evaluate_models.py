from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import linear_model
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

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
        
        self.m_listOfClassificationReports = []


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

            self.m_listOfClassificationReports.append(cr)


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

    
    ''' This is a function to manually calculate scores such as precision, recall, etc. 
        Just to be able to not rely on the classification report. A lot of calculations
        require the confusion matrix (cm) and they have 4 values a piece. For example
        give a small matrix to detect whether an image is a dog or not, the 4 values are
        defined as:
        
        True positive (tp) - The p in true positive indicates what the model predicted,
            and in this case it predicted a dog. Then the t in true positive indicates 
            what the actual ground truth is. So the model predicted a dog, and it was 
            in fact a dog due to the ground truth. They match.
            
        False positive (fp) - The p in true positive indicates what the model predicted,
            and in this case it predicted a dog. But f is false, so the ground truth 
            was NOT a dog. 
        
        True negative (tn) - The n in true negative indicates what the model predicted,
            and in this case it predicted NOT a dog. Then the t in true positive indicates 
            what the actual ground truth is. So the model predicted NOT a dog, and it was 
            in fact NOT a dog due to the ground truth. They match.

        False negative (fn) - The n in false negative indicates what the model predicted,
            and in this case it predicted NOT a dog. Then the t in true positive indicates 
            what the actual ground truth is. So the model predicted NOT a dog, and the ground
            truth shows it was in fact a dog.
        
        A confusion matrix from sklearn.metrics is shown in the form:
            TP  FN
            FP  TN

        a) Precision - Precision tells us out of ALL the positive values, in the example 
            case whether an image was a dog or not, out of all model predictions how many 
            did the model get RIGHT? Pretty standard but useful metric.

                If the model predicted dog 7 times, but only 4 were TP, that means 4 out of
                7 were correct which is 0.57. 

        b) Recall - This focuses more on the ground truths themselves, and not necessarily
            the predictions. Out of all the ground truths, how many did the model get right?

                If there are 6 dogs in all ground truths, and the model got 4 right, that means
                4 out of 6 were correct which is 0.67

        '''
    def CustomModelMetrics(self):
        for model in self.m_models:
            ''' Get classification report and confusion matrix ready to SHOW the 
                precision, recall, etc found out by the library before mathematically
                getting them. '''
            cf = classification_report(self.m_yTest, model[1].predict(self.m_xTest))
            cm = confusion_matrix(self.m_yTest, model[1].predict(self.m_xTest))
            accScore = round(accuracy_score(self.m_yTest, model[1].predict(self.m_xTest)), 2)
            print(f'{model[0]} Classification report:\n{cf}\n{model[0]} Confusion matrix:\n{cm}\nAccuracy score: {accScore}')

            # As noted in the comment above this function, confusion matrix has 4 values to help get other values.
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]

            ''' The precision and recall are explained in comment before the function. But the precision
                is pretty much a good mean, or overall performance of the model. Calculating it is 
                surprisingly simple. See the wiki page: https://en.wikipedia.org/wiki/F-score 
                
                The f1 score is denoted as:
                2 * precision * recall
                    /
                    precision + recall
                 '''
            precision = round(tp / (tp + fp), 2)
            recall = round(tp / (tp + fn), 2)
            f1 = round((2 * (precision * recall) / (precision + recall)), 2)
            print(f'Precision is {precision}')
            print(f'Recall is {recall}')
            print(f'F1 score is {f1}')
            print('\n\n')

    
    # See logistic regression predictions alone.
    def PlotRocCurves(self):
        # Get each model prediction
        yPreds = [model[1].predict(self.m_xTest) for model in self.m_models]
        #print(yPreds[0])

        # Then get the probability prediction for each model too. Use [:, 1] to only get 1 column.
        probs = [model[1].predict_proba(self.m_xTest)[:, 1] for model in self.m_models]

        # Get all auc scores.
        aucScores = [roc_auc_score(self.m_yTest, probability) for probability in probs]

        # Get all the false positive and true positive rates.
        lrFalsePosRate, lrTruePosRate, lrThreshold = roc_curve(self.m_yTest, probs[0])
        dtFalsePosRate, dtTruePosRate, dtThreshold = roc_curve(self.m_yTest, probs[1])
        rfFalsePosRate, rfTruePosRate, rfThreshold = roc_curve(self.m_yTest, probs[2])
        adaFalsePosRate, adaTruePosRate, adaThreshold = roc_curve(self.m_yTest, probs[3])

        plt.plot(lrFalsePosRate, lrTruePosRate, linestyle='--', label="Logistic Regression (AUROC = %0.3f)" % aucScores[0])
        plt.plot(dtFalsePosRate, dtTruePosRate, linestyle='--', label="Decision Tree (AUROC = %0.3f)" % aucScores[1])
        plt.plot(rfFalsePosRate, rfTruePosRate, linestyle='--', label="Random Forest (AUROC = %0.3f)" % aucScores[2])
        plt.plot(adaFalsePosRate, adaTruePosRate, linestyle='--', label="Adaboost Model (AUROC = %0.3f)" % aucScores[3])

        plt.title("LR roc plot")
        plt.xlabel("False positives")
        plt.ylabel("True positives")
        plt.legend()
        plt.show()

    
    ''' Regularization is good because it ponteitally improve the models performance. It
        penalizes high theta values. L2 regularization uses the square of the theta, but
        in L1 it uses absolute value. 

        L1 - Lasso
        L2 - Ridge  
        
        Beware, several causes of underfit models might include: Not enough data, model 
        too simple, regularization used. On the opposite side, overfitting can occur
        if the model trains on too many useless features, or if model is too complex. '''
    def ApplyRegularization(self, xTrain, yTrain):
        lm = linear_model.Lasso(alpha=50, max_iter=1000, tol=0.1)
        lm.fit(xTrain, yTrain)

        print('---Beginning Lasso and Ridge regression---')
        print(f'Lasso Model Training score: {lm.score(xTrain, yTrain)}')
        print(f'Lasso Model Testing score: {lm.score(self.m_xTest, self.m_yTest)}')

        rm = Ridge(alpha=50, max_iter=1000, tol=0.1)
        rm.fit(xTrain, yTrain)
        print(f'Ridge Model Training score: {rm.score(xTrain, yTrain)}')
        print(f'Ridge Model Testing score: {rm.score(self.m_xTest, self.m_yTest)}')
        print('---Ending Lasso and Ridge regression---\n')