#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:06:39 2018

@author: hjh
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
from sklearn.cross_validation import KFold


def process_feat():
    ad_feature=pd.read_csv('../data/adFeature.csv')
    sizeFeat =['appIdAction','appIdInstall',
               'interest1','interest2','interest3','interest4','interest5',
               'kw1','kw2','kw3',
               'topic1','topic2','topic3']
    if os.path.exists('../data/userFeature.csv'):
        user_feature=pd.read_csv('../data/userFeature.csv')
    else:
        userFeature_data = []
        with open('../data/userFeature.data', 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                    if each_list[0] in sizeFeat:
                        userFeature_dict[each_list[0] + 'size'] = len(each_list)-1
                        userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv('../data/userFeature.csv', index=False)
    train=pd.read_csv('../data/train.csv')
    predict=pd.read_csv('../data/test1.csv')
    train.loc[train['label']==-1,'label']=0
    predict['label']=-1
    data=pd.concat([train,predict])
    data=pd.merge(data,ad_feature,on='aid',how='left')
    data=pd.merge(data,user_feature,on='uid',how='left')
    
    
    beginFeat = []
    for f in sizeFeat:
        beginFeat.append(f + 'size')
    for f in beginFeat:
        data[f] = data[f].fillna(0)
    data['appAct_appInst'] = (data['appIdActionsize']/data['appIdInstallsize'])[data['appIdInstallsize'] != 0]
    data['appAct_appInst'][data['appIdInstallsize'] == 0] = 0
    data['interestNum'] = data['interest1size']
    for i in range(1,6):
        data['interestNum'] += data['interest%ssize'%i]
    for i in range(1,6):
        data['interest%sRate' % i] = (data['interest%ssize'%i] / data['interestNum'])[data['interestNum'] != 0]
        data['interest%sRate' % i][data['interestNum'] == 0] = 0

    data['kwNum'] = data['kw1size'] + data['kw2size'] + data['kw3size']
    data['kw1Rate'] = (data['kw1size'] / data['kwNum'])[data['kwNum'] != 0]
    data['kw1Rate'][data['kwNum'] == 0] = 0
    data['kw2Rate'] = (data['kw2size'] / data['kwNum'])[data['kwNum'] != 0]
    data['kw2Rate'][data['kwNum'] == 0] = 0
    data['kw3Rate'] = (data['kw3size'] / data['kwNum'])[data['kwNum'] != 0]
    data['kw3Rate'][data['kwNum'] == 0] = 0

    data['topicNum'] = data['topic1size'] + data['topic2size'] + data['topic3size']
    data['topic1Rate'] = (data['topic1size'] / data['topicNum'])[data['topicNum'] != 0]
    data['topic1Rate'][data['topicNum'] == 0] = 0
    data['topic2Rate'] = (data['topic2size'] / data['topicNum'])[data['topicNum'] != 0]
    data['topic2Rate'][data['topicNum'] == 0] = 0
    data['topic3Rate'] = (data['topic3size'] / data['topicNum'])[data['topicNum'] != 0]
    data['topic3Rate'][data['topicNum'] == 0] = 0
    data=data.fillna('-1')
    
    one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
                     'adCategoryId', 'productId', 'productType']
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
    return data, beginFeat


def LGB_predict(train_x,train_y,test_x,res, i):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=63, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=100
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res[:, i] = clf.predict_proba(test_x)[:,1]
    #res['score%i' % i] = res['score%i' % i].apply(lambda x: float('%.6f' % x))
    # res.to_csv('submission.csv', index=False)
    # os.system('zip adFeat_orgPara.zip submission.csv')
    return clf


def countVect(data, data_x, needData):
    enc = OneHotEncoder()
    one_hot_feature=['LBS','age','carrier','consumptionAbility','education',
                     'gender','house','os','ct','marriageStatus','advertiserId',
                     'campaignId', 'creativeId',
                     'adCategoryId', 'productId', 'productType']
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        data_a=enc.transform(needData[feature].values.reshape(-1, 1))
        data_x= sparse.hstack((data_x, data_a))
    print('one-hot prepared !')
    
    vector_feature=['appIdAction','appIdInstall','interest1','interest2',
                    'interest3','interest4','interest5','kw1','kw2','kw3',
                    'topic1','topic2','topic3']
    cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    for feature in vector_feature:
        cv.fit(data[feature])
        data_a = cv.transform(needData[feature])
        data_x = sparse.hstack((data_x, data_a))
    print('cv prepared !')
    return data_x


def underSampling(data, beginFeat):
    train_n = data[data.label == 0]
    train_p = data[data.label == 1]
    test=data[data.label==-1]
    res=test[['aid','uid']]
    test=test.drop('label',axis=1)
    
    
    print('underSampling')
    beginFeat.append('creativeSize')
    beginFeat.append('age')
    beginFeat.append('appAct_appInst')
    beginFeat.append('interestNum')
    beginFeat.append('interest1Rate')
    beginFeat.append('interest2Rate')
    beginFeat.append('interest3Rate')
    beginFeat.append('interest4Rate')
    beginFeat.append('interest5Rate')
    beginFeat.append('kwNum')
    beginFeat.append('kw1Rate')
    beginFeat.append('kw2Rate')
    beginFeat.append('kw3Rate')
    beginFeat.append('topicNum')
    beginFeat.append('topic1Rate')
    beginFeat.append('topic2Rate')
    beginFeat.append('topic3Rate')
    
    test_x=test[beginFeat]
    test_x = countVect(data, test_x, test)
    
    
    pre = np.zeros((test.shape[0], 20))
    
    kf = KFold(len(train_n), n_folds=20, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf):
        
        print('第{}次训练...'.format(i))
        train_ni = train_n.iloc[test_index]
        train = pd.concat([train_ni, train_p])
        train_y = train.pop('label')
        train_x = train[beginFeat]
        train_x = countVect(data, train_x, train)
        LGB_predict(train_x, train_y, test_x, pre, i)
        
        
    res['score'] = pre.mean(axis=1)
    print(res)
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('submission.csv', index=False)
    os.system('zip underSampling.zip submission.csv')
 
    

if __name__ == '__main__':
    data, beginFeat = process_feat()
    print('data shape', data.shape)
    underSampling(data, beginFeat)
        
    
    
    
    
    
    
