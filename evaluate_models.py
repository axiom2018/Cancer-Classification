from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
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



    ''' What is accuracy?
    
        It's how many of the predictions were right. Using the dog example found
        in the comment above the "PrecisionAndRecall" function, that means out
        of all the model predictions how many did the model get right? How many
        predictions correctly matched the ground truths? Doesn't matter if the
        model said dog or no dog, as long as it made the correct prediction. '''
    def Accuracy(self, showSteps=False):
        bestAccuracyModel = None

        for model in self.m_models:
            accScore = round(accuracy_score(self.m_yTest, model[1].predict(self.m_xTest)), 3)

            if showSteps is True:
                print(f'{model[0]} accuracy score is {accScore}')

            ''' Code implementations like these are there to assist in finding the best
                model based on the metric supplied by the user. For example in the BestModel
                function. '''
            if bestAccuracyModel is None:
                bestAccuracyModel = (model[0], accScore)
            else:
                if accScore > bestAccuracyModel[1]:
                    bestAccuracyModel = (model[0], accScore)

        return bestAccuracyModel




    ''' What is precision? 

        Ex: In a model that's trying to predict if a hot dog
        IS a hot dog or going for a hot dog, how many did the model get right?
            
        Out of 10 model predictions, a model can say 7 are hot dogs. But if
        only 4 are correct that gives 4/7 which is 0.57. '''
    def Precision(self, showSteps=False):
        bestPrecisionModel = None

        for model in self.m_models:
            yPred = model[1].predict(self.m_xTest)
            precScore = round(precision_score(self.m_yTest, yPred), 3)

            if showSteps is True:
                print(f"{model[0]}'s precision score: {precScore}.")

            ''' Code implementations like these are there to assist in finding the best
                model based on the metric supplied by the user. For example in the BestModel
                function. '''
            if bestPrecisionModel is None:
                bestPrecisionModel = (model[0], precScore)
            else:
                if precScore > bestPrecisionModel[1]:
                    bestPrecisionModel = (model[0], precScore)

        return bestPrecisionModel


    
    ''' What is recall? 
    
            Carrying on from the hot dog example used above, recall says "out of all
            the ground truths that are hot dogs, how many did the model actually get right?
            
            Out of 10 model predictions, if 6 are hot dogs but model got 4 right. That
            gives 4/6 which is 0.67. '''
    def Recall(self, showSteps=False):
        bestRecallModel = None

        for model in self.m_models:
            yPred = model[1].predict(self.m_xTest)
            recallScore = round(recall_score(self.m_yTest, yPred), 3)

            if showSteps is True:
                print(f"{model[0]}'s recall score: {recallScore}.")

            ''' Code implementations like these are there to assist in finding the best
                model based on the metric supplied by the user. For example in the BestModel
                function. '''
            if bestRecallModel is None:
                bestRecallModel = (model[0], recallScore)
            else:
                if recallScore > bestRecallModel[1]:
                    bestRecallModel = (model[0], recallScore)

        return bestRecallModel
    
            
            

    ''' What is roc_auc_score? 

            It's a way to calculate the auc (area under the roc curve) GIVEN a roc curve. 
            Roc curves are shown on a 2d plane where the x axis is normally the false 
            positive rate and the y axis is normally the true positive rate. Then each 
            plotting of a point means a certain confusion matrix has a certain amount of
            true and false positives. 
        
            So the auc measure the entire 2d area UNDERNEATH that curve.  '''
    def RocAucScore(self, showSteps=False):
        bestAucModel = None

        for model in self.m_models:
            yPred = model[1].predict(self.m_xTest) 
            aucScore = round(roc_auc_score(self.m_yTest, yPred), 3)

            if showSteps is True:
                print(f"{model[0]}'s auc score: {aucScore}")
        
            if bestAucModel is None:
                bestAucModel = (model[0], aucScore)
            else:
                if aucScore > bestAucModel[1]:
                    bestAucModel = (model[0], aucScore)

        return bestAucModel



    ''' What is an f1 score?
        
            It's a simple metric that finds that mean between precision and recall. 
            Also known as the "harmonic mean" '''
    def F1Score(self, showSteps=False):
        bestF1Model = None

        for model in self.m_models:
            yPred = model[1].predict(self.m_xTest) 
            f1Score = round(f1_score(self.m_yTest, yPred), 3)

            if showSteps is True:
                print(f"{model[0]}'s F1 score: {f1Score}")

            if bestF1Model is None:
                bestF1Model = (model[0], f1Score)
            else:
                if f1Score > bestF1Model[1]:
                    bestF1Model = (model[0], f1Score)

        return bestF1Model



    # Classification reports are another way to get Precision, Recall, F1 score and more.
    def ClassificationReport(self):
        for model in self.m_models:
            cr = classification_report(self.m_yTest, model[1].predict(self.m_xTest))
            print(f'{model[0]} classification report:\n{cr}')



    ''' Confusion Matrix
    
        It gives a 2 by 2 matrix of cancer (malignant or m) and not cancer (benign or b).

        The top row shows how many that were cancerous in total, left and right.

        A confusion matrix from sklearn.metrics is shown in the form:
            TN  FP
            FN  TP

        TP - Prediction is cancer, ground truth was cancer. 
        FN - Prediction is NOT cancer, ground truth was cancer.

        FP - Prediction is cancer, ground truth was not cancer.
        RN - Prediction is not cancer, ground truth was not cancer. 
        
        Use every model and get the predictions for the model on the xtest values. 
        Then compare with the first argument given in the confusion_matrix, which is
        ground truths, which are the CORRECT values. '''
    def ConfusionMatrices(self):
        for model in self.m_models:
            cm = confusion_matrix(self.m_yTest, model[1].predict(self.m_xTest))
            print(f'Confusion matrix for {model[0]} is:\n{cm}')


    
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
            TN  FP
            FN  TP

        a) Precision - Precision tells us out of ALL the positive values, in the example 
            case whether an image was a dog or not, out of all model predictions how many 
            did the model get RIGHT? Pretty standard but useful metric.

                If the model predicted dog 7 times, but only 4 were TP, that means 4 out of
                7 were correct which is 0.57. 

        b) Recall - This focuses more on the ground truths themselves, and not necessarily
            the predictions. Out of all the ground truths, how many did the model get right?

                If there are 6 dogs in all ground truths, and the model got 4 right, that means
                4 out of 6 were correct which is 0.67. '''
    def CustomModelMetrics(self):
        for model in self.m_models:
            ''' Get classification report and confusion matrix ready to SHOW the 
                precision, recall, etc found out by the library before mathematically
                getting them. '''
            cf = classification_report(self.m_yTest, model[1].predict(self.m_xTest))
            cm = confusion_matrix(self.m_yTest, model[1].predict(self.m_xTest))
            accScore = round(accuracy_score(self.m_yTest, model[1].predict(self.m_xTest)), 3)
            print(f'{model[0]} Classification report:\n{cf}\n{model[0]} Confusion matrix:\n{cm}\nAccuracy score: {accScore}')

            # As noted in the comment above this function, confusion matrix has 4 values to help get other values.
            tn = cm[0][0]
            fn = cm[1][0]
            tp = cm[1][1]
            fp = cm[0][1]

            ''' The precision and recall are explained in comment before the function. But the precision
                is pretty much a good mean, or overall performance of the model. Calculating it is 
                surprisingly simple. See the wiki page: https://en.wikipedia.org/wiki/F-score 
                
                The f1 score is denoted as:
                2 * precision * recall
                    /
                    precision + recall
                 '''
            precision = round(tp / (tp + fp), 3)
            recall = round(tp / (tp + fn), 3)
            f1 = round((2 * (precision * recall) / (precision + recall)), 3)
            print(f'Precision is {precision}')
            print(f'Recall is {recall}')
            print(f'F1 score is {f1}')
            print('\n\n')

    

    ''' Plotting roc curves are crucial for a team to see where they model is at. On the 
        x axis is normally the false positive (fp) rate or sometimes precision. The y 
        axis is the true positive (tp) rate. Each axis is from 0 to 1. 

        It's important to know the tp and fp formula:

            tp = tp / tp + fn 
            fp = fp / fp + tn
        
        Of course the confusion matrix will provide these values. What if there was a 
        confusion matrix that looked like this:

                Tp  Fp
                4   4
                0   0
                Fn  Tn

        Also it's important to know that threshold plays a huge part in the confusion
        matrix development. Playing with it's value will give a new outcome or matrix.
            
        Then plugging in the values for the tp and fp formula, they come up to:

            4 / 4 + 0 so the tp is 1
            4 / 4 + 0 so the fp is 1 as well.

        That means if this is plotted on the roc graph which both axis' are from 0 to 1,
        that tells us that the model had a lot of correct values since the true positive
        rate is so high! But it also had the same amount of false positives, which can
        be dangerous depending on the problem you're trying to tackle with the model. '''
    def PlotRocCurves(self):
        for model in self.m_models:
            # Predicting the probability of diagnosis 
            probability = model[1].predict_proba(self.m_xTest)[:, 1]

            # Get the auc score to show for the plot.
            auc_score = roc_auc_score(self.m_yTest, probability)

            ''' Fp and tp will be used on the x and y axis. Threshold is important
                because it can definitely change curves. '''
            fp, tp, threshold = roc_curve(self.m_yTest, probability)

            # Detailed string redy for the plot.
            plotLabel = f'{model[0]} (AUROC = %0.3f)' % auc_score
            plt.plot(fp, tp, linestyle='--', label=plotLabel)

        plt.title("LR roc plot")
        plt.xlabel("False positives")
        plt.ylabel("True positives")
        plt.legend()
        plt.show()

    
    ''' Helper function whose primary job is to decide which, out of all the matrices,
        has the lowest value. The BestModel function will be using this.

        modelsWithMatrices - The list comprehension does 1 thing which is get a list of 
            tuples in the form of "(name of model, confusion matrix of that model)". In 
            order to check for false positives the confusion matrices are needed.
        
        xIndex/yIndex - For example, if the BestModel function wanted to get the 
            model who has the best false positive rate, then the false positive
            would be at index 0, 1 in a matrix. So it would pass 0 and 1.
            
        metric - The name of the metric the function caller wants. '''
    def LowestValuesOfConfusionMatrix(self, xIndex, yIndex, metric, showSteps=False):
        modelsWithMatrices = [(model[0], confusion_matrix(self.m_yTest, model[1].predict(self.m_xTest))) for model in self.m_models]

        bestModel = None
            
        ''' Trying to do a list comprehension with comparisons will likely be way to unreadable so a 
            regular for loop is used here. 
            
            Getting the confusion matrix out of the tuple is necessary with the first line. Then
            if no best model (best according to the current argument passed) is selected then just
            assign the first entry as best.
            
            Then calculate the best models false positive rate. Best model is a tuple of the model
            name and the false positive, which in the confusion matrix itself, is at index 0 & 1.
            
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix 
            
            The link makes it clear that: 
                1) True negative/tn - index 0,0
                2) False negatives/fn - index 1,0
                3) True positive/tp - index 1,1
                4) False positive/fp - index 0,1 '''
        for entity in modelsWithMatrices:
            cm = entity[1]

            if showSteps is True:
                print(f'{entity[0]} has confusion matrix:\n{cm}')
            
            if bestModel is None:
                bestModel = (entity[0], entity[1][xIndex][yIndex])
            else:
                ''' Calculate the best models false positive and the current models one as well.
                    The new models false positive can be derived straight from the new confusion
                    matrix. The current best models false positive rate was already calculated, 
                    access that with index 1.
                    
                    Then only if it's LOWER, will it be considered better. False positives and
                    false negatives are not something anyone wants MORE of. '''
                rate = cm[xIndex][yIndex]

                if showSteps is True:
                    print(f'New model {entity[0]} has a {metric} of {entity[1][xIndex][yIndex]}')

                if rate < bestModel[1]:
                    bestModel = (entity[0], entity[1][xIndex][yIndex])
        
        if showSteps is True:
            print(f'\n--Best model (based on {metric}) - Name: {bestModel[0]}. {metric} rate: {bestModel[1]} ')




    ''' There are a lot of ways to determine a best model. When engineers start out,
        it's common practice for them to look at only high accuracy that determines
        whether a model is best out of a group of models. That's not really the case.
        What if the metrics show that particular model has a high false positive (fp)
        or false negative (fn)? That would be a troublesome model.
        
        With this function the best model will be selected according to the argument
        "metric".  '''
    def BestModel(self, metric='fp'):
        # List of allowed metrics to check the argument.
        listOfMetrics = ['fp', 'fn', 'accuracy', 'precision', 'recall', 'auc', 'f1']

        if metric not in listOfMetrics:
            print(f'{metric} was not in the list of metrics. List is:\n{listOfMetrics}.')
            print('Call function again with one of the provided metrics.')
            return

        # Checks for false positive in if statement, then false negative in elif statement.
        if metric is listOfMetrics[0]:
            self.LowestValuesOfConfusionMatrix(0, 1, listOfMetrics[0], True)
    
        elif metric is listOfMetrics[1]:
            self.LowestValuesOfConfusionMatrix(1, 0, listOfMetrics[1], True)

        # Accuracy.
        elif metric is listOfMetrics[2]:
            model = self.Accuracy(True)
            print(f'Model with best {listOfMetrics[2]} comes from {model[0]} with accuracy: {model[1]}')

        # Precision.
        elif metric is listOfMetrics[3]:
            model = self.Precision(True)
            print(f'Model with best {listOfMetrics[3]} comes from {model[0]} with accuracy: {model[1]}')

        # Recall.
        elif metric is listOfMetrics[4]:
            model = self.Recall(True)
            print(f'Model with best {listOfMetrics[4]} comes from {model[0]} with accuracy: {model[1]}')

        # Auc score.
        elif metric is listOfMetrics[5]:
            model = self.RocAucScore(True)
            print(f'Model with best {listOfMetrics[5]} comes from {model[0]} with accuracy: {model[1]}')

        # Fi score.
        elif metric is listOfMetrics[6]:
            model = self.F1Score(True)
            print(f'Model with best {listOfMetrics[6]} comes from {model[0]} with accuracy: {model[1]}')