# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:23:35 2017

@author: Xiaocong
"""
#!/usr/bin/python

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import RandomizedLasso
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import classification_report,f1_score,recall_score,confusion_matrix,make_scorer
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler,NearMiss
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest,f_classif

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from vecstack import stacking

import matplotlib.pylab as plt
from matplotlib import cm
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4
import seaborn as sns

################  Import Data from SAS  #################
Xy_75 = pd.read_sas('D:\od_opioid_cont1_75.sas7bdat').drop(['PAT_ID','SelectionProb','SamplingWeight'],axis=1)
Xy_test = pd.read_sas('D:\od_opioid_cont1_25.sas7bdat').drop(['PAT_ID'],axis=1)

###############    One Hot Encoding   #####################
Xy_75 = pd.get_dummies(Xy_75,columns=['Plan', 'Payer', 'Rural'],sparse=True).drop(['Plan_1.0', 'Payer_1.0', 'Rural_1.0'] ,axis=1)
Xy_test = pd.get_dummies(Xy_test,columns=['Plan', 'Payer', 'Rural'],sparse=True).drop(['Plan_1.0', 'Payer_1.0', 'Rural_1.0'] ,axis=1)

names = pd.DataFrame((Xy_75.drop(['OD_opioid'],axis=1)).columns,columns=['feature_names'])  # save a copy of feature names

X, y = Xy_75.drop(['OD_opioid'],axis=1),Xy_75['OD_opioid']
X_test, y_test = Xy_test.drop(['OD_opioid'],axis=1),Xy_test['OD_opioid']

################  Print to File  ########################
py_results = open('D:\py_results.txt','w') # change to 'a' when print to file later

py_results.write('\n \n Table: Details of Sub-samples Used in Python (75% K-fold Train-Valid set): ' + '\n')
py_results.write('\n' + '   Total Number of Patients in this Dataset (Constructed from LifeLink Cohort Design: ')
print('{:,}'.format(len(Xy_75)),file=py_results)

py_results.write('\n' + '   Number of Patients without Opioid Overdose (majority class): ')
print('{:,}'.format(len(Xy_75.loc[Xy_75['OD_opioid']==0])),file=py_results)

py_results.write('\n' + '   Number of Patients with Opioid Overdose (minority class): ')
print('{:,}'.format(len(Xy_75.loc[Xy_75['OD_opioid']==1])),file=py_results)

py_results.write('\n' + '   Number of All Categorical Features (Set #1) Dummy-coded: ')
print('{:,}'.format(len(Features)),file=py_results)
print('\n' + '========================================================================',file=py_results)

py_results.close()

#####################  Scaling  ########################
scaler = preprocessing.StandardScaler()
X_test = scaler.fit(X).transform(X_test)
X_test = pd.DataFrame(X_test,columns=X.columns)
X = scaler.fit_transform(X)
X = pd.DataFrame(X,columns=X_test.columns)

#################  Feature Selection  ##################

# Stability Selection
rlasso = RandomizedLasso(n_jobs=-1)
rlasso.fit(X,y)
ss = pd.DataFrame(sorted(zip(map(lambda x: round(x,4),rlasso.scores_),names['feature_names']),reverse=True),
                  columns=['stability','feature_name'])
ss.to_csv('D:\selectedfeatures_stabilityselection.csv')

# L1-based feature selection
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
lsvc_coef = pd.concat([pd.DataFrame(np.transpose(lsvc.coef_),columns=['L1_SVC_coef']),names],axis=1).sort_values(by='L1_SVC_coef', ascending=False)
lsvc_coef.to_csv('D:\selectedfeatures_L1_SVC.csv')

'''compare the results from the 2 feature selection algorithms,filter features if: 
       1. scored 0 from both algorithms
       2. scored 1 or close to 1 (>0.9) from Stability Selection 
          and negative from L1 linearSVC.
'''
filtered_features = np.array(['ccs_10','ccs_100','ccs_107','ccs_111e','ccs_117','ccs_118','ccs_120',
                              'ccs_123e','ccs_124e','ccs_125','ccs_130','ccs_135','ccs_138','ccs_147',
                              'ccs_156','ccs_164','ccs_165','ccs_166','ccs_167','ccs_170','ccs_173',
                              'ccs_174','ccs_175','ccs_176','ccs_198e','ccs_246','ccs_252','ccs_253',
                              'ccs_46','ccs_47','ccs_48','ccs_51','ccs_52','ccs_53','ccs_58','ccs_63',
                              'ccs_7','ccs_88','ccs_89','ccs_90','ccs_91e','ccs_92','ccs_93e','ccs_95',
                              'ccs_97','ccs_98','drug_11','drug_16','drug_17','drug_22e','drug_27',
                              'drug_28','drug_29','drug_30','drug_42','drug_43','drug_45e','drug_47',
                              'drug_55','drug_60','drug_61','drug_86','drug_9','first_MED','max_daily_med',
                              'med','mh_1','mh_2','mh_3','mh_9','num_first_opioids','opioid_1','opioid_10',
                              'opioid_11','opioid_12','opioid_2','opioid_3','opioid_6','pain_1','pain_11',
                              'pain_13','pain_14','pain_15','pain_18','pain_19','pain_2','pain_20','pain_21',
                              'pain_22','pain_3','pain_5','pain_6','pain_7','pain_9','PDMP','polydrug_5',
                              'polydrug_6','Rural_3.0','sum_cncp','sum_earlylng','sum_overlap_5','sum_overlap_6',
                              'sum_overlap_8','unq_opioid','ccs_141e','ccs_142','ccs_159','ccs_172','ccs_1e',
                              'ccs_241e','ccs_247','ccs_250','ccs_76e','ccs_9','drug_32','drug_51','drug_58',
                              'drug_74','mh_8','opioid_7','pain_10','pain_17','Plan_3.0'])
for name in list(filtered_features):
    X = X.drop([name],axis=1)
    X_test = X_test.drop([name],axis=1)
selected_features = pd.DataFrame(X.columns.values,columns=['selected_features'])

#dtrain = pd.concat([X,y],axis=1)
#dtest = pd.concat([X_test,y_test],axis=1)
    
########  Cross-Validation with Grid Search ########

# multiple scorers
scoring = {'se':  make_scorer(recall_score,labels=[1],average=None),
           'npv': make_scorer(precision_score,labels=[0],average=None),
           'sp':  make_scorer(recall_score,labels=[0],average=None),
           'ppv': make_scorer(precision_score,labels=[1],average=None)
           }
# prepare datasets
X = np.array(X)     
y = np.array(y)    
X_test = np.array(X_test)
y_test = np.array(y_test)    

######### Try XGBoost ##########
# create a parameter dictionary
xgb_param = {'gamma':[0.2],  
             'nthread':[-1], 
             'objective':['binary:logistic'],
             'learning_rate': [0.05], 
             'max_depth': [7],
             'min_child_weight': [1],
             'subsample': [0.7],
             'colsample_bytree': [0.7],
             'colsample_bylevel':[0.8],
             'n_estimators': [1000],
             'seed': [1337]
             }
 
# Re-sampling Algorithms
rus = RandomUnderSampler(random_state=42,replacement=True)              # Random under-sampling
        
X_res,y_res = rus.fit_sample(X, y)
xgb_gs = GridSearchCV(xgb.XGBClassifier(),
                      param_grid=xgb_param,
                      cv=10,
                      scoring=scoring,
                      refit='se',
                      n_jobs=-1,
                      verbose=2)
xgb_gs.fit(X_res,y_res)
y_pred = xgb_gs.predict(X_test)

py_results = open('D:\py_results.txt','a')
py_results.write('\n Split ' + str(i) + '\n' + '   Best Estimator: ')
print(str(xgb_gs.best_estimator_),file=py_results)
py_results.write('\n' + '   Best Parameters: ')
print(str(xgb_gs.best_params_),file=py_results)
py_results.write('\n' + '   Best Scores: ')
print('/n' + '   Sensitivity: '+str(recall_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
print('/n' + '   Specificity: '+str(recall_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('/n' + '   NPV: '+str(precision_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('/n' + '   PPV: '+str(precision_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
py_results.close()

# Feature Importance
mask = SelectKBest(f_classif, k = 50).fit(X_test,y_test).get_support() # dtype=boolean
feat_imp = (selected_features[mask]).reset_index(drop=True)
f_scores = pd.DataFrame(SelectKBest(f_classif, k = 50).fit(X_test,y_test).scores_[mask],columns=['f_scores'])
feat_imp_scores = pd.concat([feat_imp,f_scores],axis=1).sort_values(['f_scores'],ascending = [False]).set_index('selected_features')
'''feat_imp_scores.plot(kind='bar', title='Feature Importances',color='purple')
plt.ylabel('Feature Importance Score')'''
g = sns.barplot(y='f_scores', x=feat_imp_scores.index, data=feat_imp_scores, palette='Purples_r')
g.set(xlabel='Features',ylabel='Fisher Score',title='Feature Importance')
g.set_xticklabels(g.get_xticklabels(),rotation=90)

############ Try Random Forest ###############
rf = RandomForestClassifier(random_state=4,warm_start=True,n_jobs=-1)     
rf_param = {'max_depth':[3,4,6,10],
            'min_samples_split':[2],
            'min_samples_leaf':[1],
            'criterion':['gini'],
            'n_estimators':[1000],
            'max_features':[50,55,60]
            }
rf_gs = GridSearchCV(rf, param_grid=rf_param,
                     cv=10,
                     scoring=scoring,
                     refit='se',
                     n_jobs=-1,
                     verbose=2)
rf_gs.fit(X_res,y_res)
y_pred = rf_gs.predict(X_test)
    
py_results = open('D:\py_results.txt','a')
py_results.write('\n' + 'Random Forest: ')
py_results.write('\n' + '   Best Estimator: ')
print('\n' + '      '+str(rf_gs.best_estimator_),file=py_results)
py_results.write('\n' + '   Best Parameters: ')
print('\n' + '      '+str(rf_gs.best_params_),file=py_results)
py_results.write('\n' + '   Best Scores: ')
print('\n' + '      Sensitivity: '+str(recall_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
print('      Specificity: '+str(recall_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('      NPV: '+str(precision_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('      PPV: '+str(precision_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
print('==================================================',file=py_results)
py_results.close()

################## Try SVM ####################
# NOTE: This takes very long time!
svm_param = {'C':np.logspace(-3, 1, 5),
             'gamma':np.logspace(-5, 1, 7)}
cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
svm_gs = GridSearchCV(SVC(kernel='rbf'), # 'kernel':('rbf','linear','sigmoid','poly')
                      param_grid=svm_param,
                      cv=cv, scoring=scoring, refit='se', n_jobs=-1, verbose=2)
svm_gs.fit(X_res,y_res)
y_pred = svm_gs.predict(X_test)
 
py_results = open('D:\py_results.txt','a')
py_results.write('\n' + 'SVM: ')
py_results.write('\n' + '   Best Estimator: ')
print('\n' + '      '+str(svm_gs.best_estimator_),file=py_results)
py_results.write('\n' + '   Best Parameters: ')
print('\n' + '      '+str(svm_gs.best_params_),file=py_results)
py_results.write('\n' + '   Best Scores: ')
print('\n' + '      Sensitivity: '+str(recall_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
print('      Specificity: '+str(recall_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('      NPV: '+str(precision_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('      PPV: '+str(precision_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
print('==================================================',file=py_results)
py_results.close()

########## Stack Classifiers #############

# Initialize 1-st level models
models = [clf_gs.best_estimator_,
          rf_gs.best_estimator_,
          svm_gs.best_estimator_]

# Compute stacking features
X_train=X_res
y_train=y_res
'''def se():
    se = make_scorer(recall_score,labels=[1],average=None)
    return se'''

S_train, S_test = stacking(models, X_train, y_train, X_test,
                           regression=False, n_folds=5,stratified=True, shuffle=True, random_state=42,verbose=2,
                           metric=recall_score) # customized scorer using make_scorer does not work! why?

# Initialize 2-nd level model
model = clf_gs.best_estimator_

# Fit 2-nd level model
model = model.fit(S_train,y_train)
y_pred = model.predict(S_test)

py_results = open('D:\py_results.txt','a')
py_results.write('\n' + 'Stackings: ')
print('\n' + '      Sensitivity: '+str(recall_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
print('      Specificity: '+str(recall_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('      NPV: '+str(precision_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('      PPV: '+str(precision_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
print('==================================================',file=py_results)
py_results.close()  

########## Scikit Majority Voting ################
eclf = VotingClassifier(estimators=[('xgb',clf_gs.best_estimator_),
                                    ('rf',rf_gs.best_estimator_),
                                    ('svm',SVC(C=0.001,gamma=10.0, kernel='rbf',probability=True, random_state=42,))],voting='soft',weights=[1,1,2],flatten_transform=True)
eclf.fit(X_res,y_res)
y_pred = eclf.predict(X_test)

py_results = open('D:\py_results.txt','a')
py_results.write('\n' + 'Majority Voting: ')
print('\n' + '      Sensitivity: '+str(recall_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
print('      Specificity: '+str(recall_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('      NPV: '+str(precision_score(y_test,y_pred,average=None,labels=[0])),file=py_results)
print('      PPV: '+str(precision_score(y_test,y_pred,average=None,labels=[1])),file=py_results)
print('==================================================',file=py_results)
py_results.close()  