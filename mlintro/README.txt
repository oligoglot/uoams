
Objective:
To build a model to predict future credit defaulters by learning from past data.

Data:
A dataset containing 1000 records of customers who have availed loans from a German bank has been provided.

Features:
checking_balance - Amount of money available in account of customers (range categories) 
months_loan_duration - Duration since loan taken 
credit_history - credit history of each customers (categorical info) 
purpose - Purpose why loan has been taken (categorical info) 
amount - Amount of loan taken 
savings_balance - Balance in account (range categories) 
employment_duration - Duration of employment 
percent_of_income - Percentage of monthly income 
years_at_residence - Duration of current residence 
age - Age of customer 
other_credit - Any other credits taken (categorical info) 
Housing - Type of housing, rent or own (categorical info) 
existing_loans_count - Existing count of loans 
job - Job type (categorical info) 
dependents - Any dependents on customer  
phone - Having phone or not (boolean) 
default - Default status (Target column) 

Pre-processing and EDA:
The following pre-processing and analysis steps were performed.
1. Inspection and cleaning 
2. Converting categorical features
3. Correlation and analysis:
4. Train Test Split

Model building:
The following models were built using 80% training data and evaluated on the remaining 20% test data.
1. Logistic Regression 
2. LDA
3. QDA
4. KNN
5. Gradient Boost
6. Random Forest
7. Catboost

Conclusion:
Following are the key outcomes and observations from the exercise described above. 

    Low correlation on some features, dropping gives about the same performance. 

    Logit gives the best performance at 0.76 accuracy. CatBoost gives about the same performance. The only advantage being that we do not have to convert categorical attributes to numeric in the latter case. 

    Gradient Boosting and Random Forest perform poorly too with our fitting as we increase the depth. 

    One possible area for improvement is in the encoding of categorical features.  

    Credit History - We are not sure how the difference between critical and poor is vs very good and perfect.  

    Employment Duration - Can we bucket better than the current buckets?  

    Checking and Savings Balance - These should ideally have a high predictive power when used as a numeric ordinal feature. However, I've made them binary encoded because of unknown values. If we had more training data, we could try imputing the unknowns. 

    A relatively low LDA f1-score on training data suggests underfitting. Given this, we could perhaps try Target Encoder for a few of the categorical features. 

    We could also make cross-validation based improvements. 

Code and detailed doc attached.
