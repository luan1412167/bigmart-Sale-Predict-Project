import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#Read train file and test file :
train = pd.read_csv("C:/Users/luan1/Desktop/helloworldpython/bigmartsalesdata/Train.csv")
test = pd.read_csv("C:/Users/luan1/Desktop/helloworldpython/bigmartsalesdata/Test.csv")

#step 1 : pre-processing
# Combine train data and test data
train['source']='train'
test['source']='test'
data=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,data.shape)

#Check the percentage of null values per variable
print(data.isnull().sum()/data.shape[0]*100) 

#aggfunc is mean by default! Ignores NaN by default
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
print(item_avg_weight)

#data[:][data[‘Item_Identifier’] == ‘DRI11’]

def impute_weight(cols):
    Weight = cols[0]
    Identifier = cols[1]
    
    if pd.isnull(Weight):
        return item_avg_weight['Item_Weight'][item_avg_weight.index == Identifier]
    else:
        return Weight

print ('Orignal #missing: %d'%sum(data['Item_Weight'].isnull()))
data['Item_Weight'] = data[['Item_Weight','Item_Identifier']].apply(impute_weight,axis=1).astype(float)
print ('Final #missing: %d'%sum(data['Item_Weight'].isnull()))

#outlet_size
#Import mode function:
from scipy.stats import mode
#Determing the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
print(outlet_size_mode)
#replace the missing outlet_size
def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size
print ('Orignal #missing: %d'%sum(data['Outlet_Size'].isnull()))
data['Outlet_Size'] = data[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)
print ('Final #missing: %d'%sum(data['Outlet_Size'].isnull()))

# visibility =0 replace =mean() 
visibility_item_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
print(visibility_item_avg)

def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visibility_item_avg['Item_Visibility'][visibility_item_avg.index == item]
    else:
        return visibility
print ('Original #zeros: %d'%sum(data['Item_Visibility'] == 0))
data['Item_Visibility'] = data[['Item_Visibility','Item_Identifier']].apply(impute_visibility_mean,axis=1).astype(float)
print ('Final #zeros: %d'%sum(data['Item_Visibility'] == 0))


# replace establish year into working year
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year'] ## data have from 2013
data['Outlet_Years'].describe()

#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})                                                                                                     
print(data['Item_Type_Combined'].value_counts())

#Change categories of low fat:
print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())
print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())

func = lambda x: x['Item_Visibility']/visibility_item_avg['Item_Visibility'][visibility_item_avg.index == x['Item_Identifier']][0]
data['Item_Visibility_MeanRatio'] = data.apply(func,axis=1).astype(float)
print(data['Item_Visibility_MeanRatio'].describe())

#Import labelEncoder library : 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    data[i] = le.fit_transform(data[i])

#Dummy Variables:
data = pd.get_dummies(data, columns =['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
print(data.dtypes)
#print(data.Outlet_Size_0,data.Outlet_Size_1,data.Outlet_Size_2)

#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]
#print(train['source'])
#Drop unnecessary columns:
train.drop(['source'],axis=1,inplace=True,errors='ignore')
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True,errors='ignore')

#Export files as modified versions:
train.to_csv("C:/Users/luan1/Desktop/helloworldpython/bigmartsalesdata/train_modified.csv",index=False)
test.to_csv("C:/Users/luan1/Desktop/helloworldpython/bigmartsalesdata/test_modified.csv",index=False)

train_df = pd.read_csv('C:/Users/luan1/Desktop/helloworldpython/bigmartsalesdata/train_modified.csv')
test_df = pd.read_csv('C:/Users/luan1/Desktop/helloworldpython/bigmartsalesdata/test_modified.csv')

# print(train_df[0:1][0:32].T)
# print(test_df)


# #Define target and ID columns:
# target = 'Item_Outlet_Sales'
# IDcol = ['Item_Identifier','Outlet_Identifier']

# from sklearn import cross_validation, metrics

# def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
#     #Fit the algorithm on the data
#     alg.fit(dtrain[predictors], dtrain[target])
        
#     #Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
    
#     #Remember the target had been normalized
#     Sq_train = (dtrain[target])**2
#     #Perform cross-validation:
#     cv_score = cross_validation.cross_val_score(alg, dtrain[predictors],Sq_train , cv=20, scoring='neg_mean_squared_error')
#     cv_score = np.sqrt(np.abs(cv_score))
    
#     #Print model report:
#     print("\nModel Report")
#     print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(Sq_train.values, dtrain_predictions)))
#     print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

#     #Export submission file:
#     IDcol.append(target)
#     submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
#     submission.to_csv(filename, index=False)

# # from sklearn.linear_model import LinearRegression
# # LR = LinearRegression(normalize=True)
# predictors = train_df.columns.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])

# # print(predictors[0:2][0:32])
# # modelfit(LR, train_df, test_df, predictors, target, IDcol, 'LR.csv')
# from sklearn.tree import DecisionTreeRegressor
# DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
# modelfit(DT, train_df, test_df, predictors, target, IDcol, 'DT.csv')

mean_sales = train_df['Item_Outlet_Sales'].mean()

baseline_submission = pd.DataFrame({
'Item_Identifier':test_df['Item_Identifier'],
'Outlet_Identifier':test_df['Outlet_Identifier'],
'Item_Outlet_Sales': mean_sales
},columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])
print(baseline_submission)

from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
X_train = train_df.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'],axis=1)
Y_train = train_df['Item_Outlet_Sales']
X_test = test_df.drop(['Item_Identifier','Outlet_Identifier'],axis=1).copy()
lr.fit(X_train, Y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = round(lr.score(X_train,Y_train) * 100,2)
print('sai so la %.4g' %lr_accuracy)

#submission
linear_submission = pd.DataFrame({
'Item_Identifier':test_df['Item_Identifier'],
'Outlet_Identifier':test_df['Outlet_Identifier'],
'Item_Outlet_Sales': lr_pred
},columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])

linear_submission.to_csv('linear_algo.csv',index=False)

#Decision tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=15,min_samples_leaf=100)
tree.fit(X_train,Y_train)
tree_pred = tree.predict(X_test)
tree_accuracy = round(tree.score(X_train,Y_train)*100,2)
print('sai so dicision la : %.4g'%tree_accuracy)

tree_submission = pd.DataFrame({
'Item_Identifier':test_df['Item_Identifier'],
'Outlet_Identifier':test_df['Outlet_Identifier'],
'Item_Outlet_Sales': tree_pred
},columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])

tree_submission.to_csv('tree_algo.csv',index=False)

#randomForest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
rf.fit(X_train,Y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = round(rf.score(X_train,Y_train)*100,2)
print('sai so randomforest la : %.4g' %rf_accuracy)