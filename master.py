#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import pandas as pd


# In[120]:


data = pd.read_csv("masterdata_prediction.csv")


# In[121]:


data.head()


# In[122]:


data.info()


# In[123]:


sap=data.columns


# In[124]:


new_col = []
for i in sap:
    new_col.append(str(i))


# In[125]:


deal=new_col[0].split(";")


# In[126]:


deal


# In[128]:


import string
import re


# In[129]:


chars = re.escape(string.punctuation)


# In[130]:


chars


# In[131]:


richy= []
for i in deal:
    sancho = re.sub(r'['+chars+']', '',i)
    richy.append(sancho)


# In[132]:


richy


# In[133]:


new = data.iloc[:,0].str.split(pat=";",expand=True)


# In[134]:


new


# In[135]:


new.set_axis(richy, axis=1,inplace=True)


# In[136]:


new.head()


# In[137]:


new.set_index("id",inplace=True)


# In[138]:


new.head(5)


# In[21]:


new.info()


# In[139]:


new.merchantpayout.head(3)


# In[140]:


new.replace("",np.nan,inplace=True)


# In[141]:


new.columns


# In[142]:


new.info()


# In[143]:


train=new[['transactionoriginationcountry', 'bankid', 'pricepoint', 'currencycode',
     'date', 'status', 'merchantid', 'executiontype','createddate',
     'lastmodifieddate', 'createdday', 'createdhour','startdevice', 'enddevice','bankname']]


# In[144]:


train.head(3)


# In[145]:


def cleanData(x):
    tyrell = str(x)
    sancho = re.sub(r'['+chars+']', '',tyrell)
    return sancho


# In[146]:


train["transactionoriginationcountry"] = train["transactionoriginationcountry"].apply(cleanData)


# In[147]:


train["bankid"] = train["bankid"].apply(cleanData)


# In[148]:


train["currencycode"] = train["currencycode"].apply(cleanData)
train["status"] = train["status"].apply(cleanData)
train["executiontype"] = train["executiontype"].apply(cleanData)
train["createdhour"] = train["createdhour"].apply(cleanData)
train["startdevice"] = train["startdevice"].apply(cleanData)
train["bankname"] = train["bankname"].apply(cleanData)


# In[149]:


train.head(3)


# In[150]:


train["enddevice"] = train["enddevice"].apply(cleanData)


# In[151]:


train.drop("createddate",axis=1,inplace=True)


# In[152]:


train.info()


# In[153]:


train.head(3)


# In[154]:


train.tail(3)


# In[155]:


def getDayT(a):
    x= a.split("-")
    return x[2]


# In[156]:


train["date"]= train["date"].apply(getDayT)


# In[157]:


train["lastmodifieddate"]= train["lastmodifieddate"].apply(getDayT)


# In[158]:


train.head()


# In[159]:


train["date"][1]


# In[160]:


def getDay(a):
    x= a.split(" ")
    return x[0]


# In[161]:


train["day"]= train["date"].apply(getDay)


# In[162]:


train.head(3)


# In[163]:


train.drop("createdday",axis=1,inplace=True)


# In[164]:


def getTime(a):
    x= a.split(" ")
    return x[1]


# In[165]:


train["time"]=train["date"].apply(getTime)


# In[166]:


train.drop("date",axis=1, inplace=True)


# In[167]:


train["lastmodifiedtime"]=train["lastmodifieddate"].apply(getTime)


# In[168]:


train.tail(5)


# In[169]:


train[train["createdhour"] == ""]


# In[170]:


train.drop(["time","lastmodifiedtime","lastmodifieddate"],axis=1,inplace=True)


# In[171]:


train.replace("",np.nan,inplace=True)


# In[172]:


train["createdhour"].fillna("23",inplace=True)


# In[173]:


train["bankid"]= train["bankid"].astype(int)
train["pricepoint"]= train["pricepoint"].astype(float)
train["createdhour"]= train["createdhour"].astype(int)
train["day"]= train["day"].astype(int)


# In[174]:


train.info()


# In[175]:


train["merchantid"]= train["merchantid"].astype(int)


# In[176]:


train.head()


# In[177]:


train["executiontype"].value_counts()


# In[178]:


train["currencycode"].value_counts()


# In[179]:


train["status"].value_counts()


# In[180]:


train["startdevice"].value_counts()


# In[181]:


train["enddevice"].value_counts()


# In[182]:


train["startdevice"].fillna("MOBILE",inplace=True)
train["enddevice"].fillna("enddevice",inplace=True)


# In[183]:


train['startdevice'] = train['startdevice'].map(train['startdevice'].value_counts().to_dict())
train['enddevice'] = train['enddevice'].map(train['enddevice'].value_counts().to_dict())
train['status'] = train['status'].map(train['status'].value_counts().to_dict())
train['bankname'] = train['bankname'].map(train['bankname'].value_counts().to_dict())


# In[184]:


train = pd.get_dummies(train, columns = ["currencycode"])


# In[185]:


train.drop(["executiontype"],axis=1,inplace=True)


# In[186]:


train.info()


# In[187]:


X = train.drop('pricepoint', axis=1)
y = train.pricepoint


# In[188]:


from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


# In[189]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[195]:


country=X_test["transactionoriginationcountry"]


# In[197]:


X_test.drop("transactionoriginationcountry",axis=1,inplace=True)
X_train.drop("transactionoriginationcountry",axis=1,inplace=True)


# In[190]:


from sklearn.metrics import mean_squared_error


# In[191]:


def metric(x,y):
    return np.sqrt(mean_squared_error(x,y))


# In[192]:


from catboost import CatBoostRegressor


# In[193]:


model=CatBoostRegressor(iterations=2300,
    learning_rate=0.1,
    depth=8,random_strength=10,l2_leaf_reg=4,bagging_temperature=0.5)


# In[198]:


model.fit(X_train,y_train)


# In[199]:


pred1=model.predict(X_test)


# In[200]:


metric(y_test,pred)


# In[202]:


sub2= pd.DataFrame()
sub2["merchantid"] = X_test["merchantid"]
sub2["day"] = X_test["day"]
sub2["country"] = country
sub2["Original_Price_Point"]= y_test
sub2["Predicted_PricePoint"] = np.round(pred1)
sub2.to_csv("Predictions2.csv")


# In[204]:


sub2.head()

