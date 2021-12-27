#!/usr/bin/env python
# coding: utf-8

# In[441]:


# Here we are importing all the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE


# In[14]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv('Housing.csv')


# In[5]:


df.head()


# In[6]:


df.info()
# We can clearly see that there are no null values and all the columns are in right format


# In[7]:


# Dataset doesn't contain any null values
df.isna().sum()


# In[8]:


df.describe()


# In[ ]:


### Here we are visualizing our data to gain some insights from it.


# In[19]:


# Here we are trying to see the correlation b/w all the numerical variables and based on that we can clearly see that area and bathrooms has some coorelation with price
plt.figure(figsize = (10,8))
sns.heatmap(df.corr() , annot = True  )


# In[13]:


sns.pairplot(df)


# In[20]:


df.columns


# In[23]:


df.select_dtypes(exclude=['int64']).columns.tolist()


# In[ ]:


# Let's plot boxplot for categorical variables


# In[36]:


plt.figure(figsize=(16 , 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'mainroad'  , y = 'price' , data = df)
plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom'  , y = 'price' , data = df)
plt.subplot(2,3,3)
sns.boxplot(x = 'basement'  , y = 'price' , data = df)
plt.subplot(2,3,4)
sns.boxplot(x = 'hotwaterheating'  , y = 'price' , data = df)
plt.subplot(2,3,5)
sns.boxplot(x = 'airconditioning'  , y = 'price' , data = df)
plt.subplot(2,3,6)
sns.boxplot(x = 'furnishingstatus'  , y = 'price' , data = df)
plt.show()


# In[46]:


plt.figure(figsize =  (12,6))
plt.subplot(1,2,1)
sns.boxplot(x = 'furnishingstatus' , y = 'price' , hue =  'airconditioning' , data = df )
plt.subplot(1,2,2)
sns.boxplot(x = 'furnishingstatus' , y = 'price' , hue =  'basement' , data = df )


# In[51]:


df['hotwaterheating'].replace(('yes' , 'no'),(1,0) , inplace = True )


# In[54]:


df['mainroad'].replace(('yes' , 'no'),(1,0) , inplace = True )
df['guestroom'].replace(('yes' , 'no'),(1,0) , inplace = True )
df['basement'].replace(('yes' , 'no'),(1,0) , inplace = True )
df['airconditioning'].replace(('yes' , 'no'),(1,0) , inplace = True )
df['prefarea'].replace(('yes' , 'no'),(1,0) , inplace = True )


# In[49]:


df.hotwaterheating.value_counts()


# In[475]:


#Converted every categoriccal variables into binary


# In[57]:


status = pd.get_dummies(df['furnishingstatus'])


# In[60]:


status = pd.get_dummies(df['furnishingstatus'] , drop_first = True)


# In[66]:


df = pd.concat([df , status] , axis = 1)


# In[70]:


df.drop('furnishingstatus' ,axis =1 , inplace = True)


# In[ ]:





# ####  -Now we have converted all the categorical variables into numerical variables 
# 

# #### **** Lets's split our dataset into trainig and test set , for training and testing of the model. ****

# In[77]:


X = df.drop('price' , axis = 1)


# In[81]:


y = df.iloc[: , :1]


# In[145]:


# Now we are splitting our dataset into training and testing set.

X_train , X_test , y_train , y_test = train_test_split(X, y ,test_size = 0.2)


# In[133]:


Df_train , df_test =  train_test_split(df , test_size = 0.25)


# In[134]:


# Here we are creating the instance of MinMaxScaler for scaling the dataset
scaling =MinMaxScaler()


# #### Here we are converting every variable on same scale by using MinMaxScaling

# In[105]:


cols = ['price' , 'area' , 'bedrooms' , 'bathrooms' ,'stories']


# In[135]:


Df_train[cols] = scaling.fit_transform(Df_train[cols])


# In[136]:


plt.figure(figsize = (16 , 10))
sns.heatmap(Df_train.corr() , annot = True , cmap="YlGnBu")


# In[137]:


plt.scatter(Df_train.area , Df_train.price )
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('AREA VS PRICE')
plt.show()


# In[248]:


test_scale_columns = ['area' , 'bedrooms' , 'bathrooms' , 'stories']


# In[139]:


y_train = Df_train.pop('price')


# In[140]:


X_train = Df_train


# In[252]:


X_test[test_scale_columns] = scaling.fit_transform(X_test[test_scale_columns])


# In[257]:


y_test['price'] = scaling.fit_transform(y_test[['price']])


# In[ ]:





# In[163]:


scale_columns = ['area' , 'bedrooms' , 'bathrooms' , 'stories']


# In[165]:


X_train[scale_columns] = scaling.fit_transform(X_train[scale_columns])


# In[185]:


y_train[['price']] = scaling.fit_transform(y_train[['price']])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Building model with only one variable

# In[202]:


ar = X_train.iloc[: ,  :1]


# In[203]:


ar_model = LinearRegression()


# In[204]:


ar_model.fit(ar , y_train)


# In[205]:


ar_model.intercept_


# In[207]:


ar_model.coef_


# In[211]:


plt.scatter(ar , y_train)
plt.xlabel('Area')
plt.ylabel('Price')
plt.plot(ar , 0.12284002 + 0.56940017*ar , 'r')


# In[ ]:


val = X_test.iloc[: , :1]


# In[271]:


predicted_val = ar_model.predict(val)


# In[279]:


plt.scatter(y_test , val)
plt.plot(val , predicted_val , 'r')


# In[281]:


r2_score(y_test , predicted_val)


# ### Now let's create a model with two variables 

# In[225]:


# X_train
two_vars = X_train[['area' , 'bathrooms']]


# In[294]:


# X_test
tw_lr_X_test = X_test[['area' , 'bathrooms']]


# In[226]:


tw_lr = LinearRegression()


# In[227]:


tw_lr.fit(two_vars , y_train)


# In[229]:


tw_lr.coef_


# In[231]:


tw_lr.intercept_


# In[286]:


y_test.shape


# In[295]:


tw_lr_predicted = tw_lr.predict(tw_lr_X_test)


# In[ ]:





# In[296]:


r2_score(y_test , tw_lr_predicted)


# #### We can clearly see that the value of r2_score increased after adding the variables

# In[297]:


mean_squared_error(y_test , tw_lr_predicted)


# ### No we are making model by adding all the variables

# In[302]:


all_vars_model = LinearRegression()


# In[304]:


all_vars_model.fit(X_train , y_train)


# In[306]:


all_vars_model.coef_


# In[307]:


all_vars_model.intercept_


# In[309]:


X_test_predicted = all_vars_model.predict(X_test)


# In[310]:


r2_score(X_test_predicted , y_test)


# In[311]:


plt.scatter(X_test_predicted , y_test)


# ## Now we are trying to build our model with statsmodels library.

# #### First we build model with all the variabless  , and after looking at the different evaluation metrics we keep dropping the insignificant variables inorder to increase the performance of the model

# In[313]:


X_train_lm = sm.add_constant(X_train)


# In[320]:


sm_lr = sm.OLS(y_train  , X_train_lm).fit()


# In[321]:


sm_lr.params


# In[326]:


sm_lr.summary()


# In[341]:


# We are trying to see the VIF of the variables

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values , i) for i in range(X_train.shape[1])]
vif = round(vif , 2)
vif = vif.sort_values(by = 'VIF' , ascending = False)


# In[342]:


vif


# ### ****Here we are deleting the row with very high p-values i.e, 'Bedrooms ****

# In[347]:


updated_X_train = X_train_lm.drop('bedrooms' , axis = 1)


# In[349]:


upd_sm_model = sm.OLS(y_train , updated_X_train).fit()


# In[350]:


upd_sm_model.params


# In[351]:


upd_sm_model.summary()


# In[378]:


Vif = pd.DataFrame()
Vif['Features'] = updated_X_train.columns
Vif['VIF'] = [variance_inflation_factor(updated_X_train.values , i) for i in range(updated_X_train.shape[1])]
Vif = round(vif , 2)
Vif = Vif.sort_values(by = 'VIF' , ascending = False)


# In[379]:


Vif


# ## Here we are tying to remove the column i.e, 'mainroad'

# In[357]:


upd_X_train = updated_X_train.drop('mainroad' , axis = 1)


# In[360]:


upd_sm =  sm.OLS(y_train , upd_X_train).fit()


# In[362]:


upd_sm.params


# In[363]:


upd_sm.summary()


# In[380]:


Vvif = pd.DataFrame()
Vvif['Features'] = upd_X_train.columns
Vvif['VIF'] = [variance_inflation_factor(upd_X_train.values , i) for i in range(upd_X_train.shape[1])]
Vvif = round(Vvif , 2)
Vvif = Vvif.sort_values(by = 'VIF' , ascending = False)


# In[381]:


Vvif


# In[ ]:





# ## Now we are dropping 'Semi-Furnished' column 

# In[384]:


x = upd_X_train.drop('semi-furnished' , axis = 1)


# In[387]:


unfur_model = sm.OLS(y_train , x).fit()


# In[389]:


unfur_model.params


# In[390]:


unfur_model.summary()


# In[391]:


Vvvif = pd.DataFrame()
Vvvif['Features'] = x.columns
Vvvif['VIF'] = [variance_inflation_factor(x.values , i) for i in range(x.shape[1])]
Vvvif = round(Vvif , 2)
Vvvif = Vvif.sort_values(by = 'VIF' , ascending = False)


# In[393]:


Vvvif


# #### Now we are good to go with this model , because all the varibles have very small p-value and the value of VIF is also less than 5

# In[394]:


unfur_model


# In[397]:


X_test = sm.add_constant(X_test)


# In[408]:


predicted_y = unfur_model.predict(X_test)


# In[405]:


x.shape


# In[406]:


X_test = X_test.drop(['bedrooms' , 'mainroad' , 'semi-furnished' ] , axis = 1)


# In[435]:


sns.distplot(y_test - predicted_y)


# In[419]:


plt.figure(figsize = (8,6))
plt.scatter(predicted_y , y_test)
plt.xlabel('Predicted Y' , fontsize = 15)
plt.ylabel('Y_test' , fontsize = 15)
plt.title('Y_predicted VS Y_test' , fontsize = 20)
plt.show()


# In[422]:


unfur_model.params


# In[428]:


unfur_model.summary()


# ### Now we created the model ,accuracy is not too good but it's fine.

# In[ ]:


price = area *0.323844 + bathrooms * 0.268666  + stories * 0.126261 + guestroom * 0.029472 + basement * 0.025465 + hotwaterheating * 0.076315 + airconditioning * 0.078979 + parking * 0.024124 + prefarea * 0.056786 + unfurnished * -0.032490


# ### We can predict the price of house with the help of above formula

# ## Now we are trying to build model using RFE

# In[439]:


linear_model = LinearRegression()


# In[440]:


linear_model.fit(X_train , y_train)


# In[445]:


rfe_model = RFE(linear_model , 10)
rfe_model = rfe_model.fit(X_train , y_train)


# In[448]:


list(zip(X_train.columns , rfe_model.support_ , rfe_model.ranking_))


# In[452]:


cols = X_train.columns[rfe_model.support_]


# In[457]:


cols = X_train[cols]


# In[459]:


sm_x_train = sm.add_constant(cols)


# In[467]:


sm_model = sm.OLS( y_train ,  sm_x_train).fit()


# In[468]:


sm_model.summary()


# In[472]:


updated_sm_x_train = sm_x_train.drop('guestroom' , axis = 1)


# In[473]:


lrrr_model = sm.OLS(y_train , updated_sm_x_train).fit()


# In[474]:


lrrr_model.summary()


# #### The formla for predicting the price of the house - 

# price = area *0.323844 + bathrooms * 0.268666  + stories * 0.126261 + guestroom * 0.029472 + basement * 0.025465 + hotwaterheating * 0.076315 + airconditioning * 0.078979 + parking * 0.024124 + prefarea * 0.056786 + unfurnished * -0.032490

# In[ ]:




