'''

            Data 

This is a base class that will be used to implement functions that all derived classes will override. 
Since all classes will need to perform certain operations, this makes sense to do.

    Interface:

    1) Display - Used for streamlit, this function will call functions such as st.write() and more.

    2) UpdateDataframe - For a class like DataAnalysis.py, it's not really necessary to update the
        dataframe. However for DataCleaningAndValidation.py, it MIGHT be necessary because of 
        outlier removal. In that case then it's conditional. The condition is did the user
        click the checkbox to show they are WILLING to removal outliers? If so, that affects the
        dataframe and DataCleaningAndValidation.py has to figure out whether to return the
        regular dataframe or the edited one without outliers.

    
    Classes that will implement all/mix of functions:

    1) DataAnalysis - Will use Display function, no need for UpdateDataframe.

    2) DataCleaningAndValidation - Will use both since outlier removal might
        happen.
    

'''

class Data:
    def Display(self):
        pass
    
    def UpdateDataframe(self):
        pass