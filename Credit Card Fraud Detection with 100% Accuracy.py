#!/usr/bin/env python
# coding: utf-8

# # 1.Importing the libraries

# In[2]:


dat


# # 2. Data Acquisition and Description

# In[3]:


# Let load our data
data = pd.read_csv('C:/Users/black/Downloads/archive/creditcard.csv')


# In[4]:


print ('Shape of our Dataset -', data.shape)
data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# Observations
# - We have 284807 rows of observations and 31 columns.
# - Class is our output feature indicating whether the transaction is fraudulent (1) or not (0).
# - No missing values observed in our dataset.
# - dtype of all the features looks perfect.

# # 3. Data Preprocessing

# In[7]:


# let check for missing values
data.isna().sum()


# In[8]:


# Let check for duplicates 
data.duplicated().any()


# In[9]:


data.duplicated()


# In[10]:


data.info()


# Observations
#  - No missing values
#  - No duplicates.
#  - dtype also looks fine.

# # Exploratory Data Analysis

# In[11]:


# Let look at the Heatmap First
paper = plt.figure(figsize=[20, 10])
sns.heatmap(data.corr(), cmap='crest',annot=True)
plt.show()


# Observations
# 
#  - Few features have high co-relation amoung different features.
#  - V17 and V18 are highly co-related.
#  - V16 and V17 are highly co-related
#  - V14 has a negative corelation with V4.
#  - V12 is also negatively co-related with V10 and V11.
#  - V11 is negatively co-related with V10 and positively co-related with V4.
#  - V3 is positively co-related with V10 and V12.
#  - V9 and V10 are also positively co-related.

# In[12]:


# Let look at the distribution using pairplot
#sns.pairplot(data=data, hue='Class')


# Observations
#    
#   - The amount is almost normally distributed.

# In[13]:


# Let skew the skewness of our features
data.skew()


# Observations
# 
# - Features like V1, V10, V23 are highly negatively skewed.
# - Let see the distribution of some of these features.

# In[14]:


# let see the distribution of 'amount feature'
data['Amount'].plot.box()


# In[15]:


sns.kdeplot(data=data['Amount'], shade=True)
plt.show()


# Observations:
# - Amount is fairly normally distributed.

# In[16]:


# Let plot a histogram
paper, axes = plt.subplots(2, 2, figsize=(10,6))
data['V1'].plot(kind='hist', ax=axes[0,0], title='Distribution of V1')
data['V10'].plot(kind='hist', ax=axes[0,1], title='Distribution of V10')
data['V12'].plot(kind='hist', ax=axes[1,0], title='Distribution of V12')
data['V23'].plot(kind='hist', ax=axes[1,1], title='Distribution of V23')
plt.suptitle('Distribution of V1, V10, V12, and V23', size=14)
plt.tight_layout()
plt.show()


# In[17]:


# Let look at out Output feature
data['Class'].value_counts().plot.pie(explode=[0.1,0], autopct='%3.1f%%' ,shadow=True, legend=True,startangle=45)
plt.title('Distribution of Class', size=14)
plt.show()


# Observations
# 
# - The output feature is equally balanced.

# # 4. Data Preparation

# In[18]:


# Let prepare the data for the Model
data.head()


# In[19]:


# Let divide the data into dependent and independent features
x = data.drop(['Time', 'Class'], axis=1)
y = data.Class


# In[20]:


x.head()


# In[21]:


print('Shape of x',x.shape)
print('Shape of y',y.shape)


# In[22]:


# Let standardize all the features to bring them on the same scale.
sc =StandardScaler()


# In[23]:


x_scaled = sc.fit_transform(x)


# In[24]:


x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)


# In[25]:


x_scaled_df.head()


# # 5. Modelling

# In[26]:


# Let Split the dataset into train and test
x_train,x_test,y_train,y_test = train_test_split(x_scaled_df,y,test_size=0.25,random_state=15,stratify=y)


# In[27]:


# Let see the shapes
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# Logistic Regression

# In[28]:


# Let build a Logistic Regression Model
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[29]:


# Let define a function for Checking Model Accuracy, Classification Report and Confussion Matrix
def model_eval(actual, predicted):
    acc_score = accuracy_score(actual, predicted)
    conf_matrix = confusion_matrix(actual, predicted)
    clas_rep = classification_report(actual, predicted)
    print('Model Accuracy is: ', round(acc_score, 2))
    print(conf_matrix)
    print(clas_rep)


# In[30]:


preds_lr_train = lr.predict(x_train)
preds_lr_test = lr.predict(x_test)


# In[31]:


# Let see the Evaluation matrix of train and test dataset
print('-------Training Accuracy---------')
model_eval(y_train,preds_lr_train)


# In[32]:


print('-------Test Accuracy---------')
model_eval(y_test, preds_lr_test)


# Observations
# - The Logistic Regression Model is giving 100% Accuracy.
# - Let see tree-based models.

# Decision Tree

# In[33]:


# Let build a DecisionTree Model and fit
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)


# In[34]:


preds_dtree_train = dtree.predict(x_train)
preds_dtree_test = dtree.predict(x_test)


# In[35]:


print('-------Training Accuracy---------')
model_eval(y_train,preds_dtree_train)


# In[36]:


print('-------Test Accuracy---------')
model_eval(y_test,preds_dtree_test)


# Random Forest

# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


# Let build a Random Forest Classifier Model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)


# In[39]:


preds_rf_train = rf.predict(x_train)
preds_rf_test = rf.predict(x_test)


# In[40]:


print('-------Training Accuracy---------')
model_eval(y_train, preds_rf_train)


# In[41]:


print('-------Test Accuracy---------')
model_eval(y_test, preds_rf_test)


# Observations
# - Random Forest with default parameters are giving 100% accuracy on both test and train dataset.

# In[42]:


get_ipython().system('pip install xgboost')

import xgboost as xgb


# In[43]:


xgclf = xgb.XGBRFClassifier()
xgclf.fit(x_train,y_train)


# In[44]:


preds_xgb_train = xgclf.predict(x_train)
preds_xgb_test = xgclf.predict(x_test)


# In[45]:


print('-------Training Accuracy---------')
model_eval(y_train,preds_xgb_train)


# In[46]:


print('-------Test Accuracy---------')
model_eval(y_test,preds_xgb_test)


# Hypertuning

# In[47]:


# Let try to do some hyperparameter tuning to select the best parameters
from sklearn.model_selection import RandomizedSearchCV


# In[48]:


# Hyperparameter tuning for XGBoost
param_dist_xgb = {
    'n_estimators': [50,100,150,200,300,400],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6]
}


# In[49]:


xgb_clf = RandomizedSearchCV(xgclf,param_dist_xgb, verbose = 2)


# In[50]:


xgb_clf.fit(x_train,y_train)


# In[51]:


# Best Hyper Parameters for XG Boost
print('Best Paramaters for XG Boost :', xgb_clf.best_params_)


# In[52]:


preds_xgb_clf_train = xgb_clf.predict(x_train)
preds_xgb_clf_test = xgb_clf.predict(x_test)


# In[53]:


print('-------Training Accuracy--------')
model_eval(y_train,preds_xgb_clf_train)


# In[54]:


print('-------Test Accuracy---------')
model_eval(y_test,preds_xgb_clf_test)


# Conclusion
# 
# - I have done Exploratory Data Analysis for different features.
# - I prepared the Data and build different ML Models.
# - I have seen how different models are performing with Accuracy, Precision, Recall and F1 Scores.
# - Random Forest with default parameters is giving 100% accuracy on training and test dataset.
# - I have tried using Boosting technique XGBoost and have a model with 100% accuracy with improvement in False Positive and False Negative.
# - I have futher tried doing hyper parameter tuning for XGBoost. 

# In[ ]:




