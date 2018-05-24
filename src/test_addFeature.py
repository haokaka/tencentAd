# coding=utf-8
# @author:bryan
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

ad_feature=pd.read_csv('../data/adFeature.csv')

sizeFeat =['appIdAction','appIdInstall',
           'interest1','interest2','interest3','interest4','interest5',
           'kw1','kw2','kw3',
           'topic1','topic2','topic3']
if os.path.exists('../data/userFeature.csv'):
    user_feature=pd.read_csv('../data/userFeature.csv')
else:
    userFeature_data = []
    with open('../data/userFeat_head10.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                if each_list[0] in sizeFeat:
                    userFeature_dict[each_list[0] + 'size'] = len(each_list)
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
    data['interest%sRate'] = (data['interest%ssize'%i] / data['interestNum'])[data['interestNum'] != 0]
    data['interest%sRate'][data['interestNum'] == 0] = 0

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
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train=data[data.label!=-1]
train_y=train.pop('label')
# train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)
enc = OneHotEncoder()

beginFeat.append('creativeSize')
beginFeat.append('age')
train_x=train[beginFeat]
test_x=test[beginFeat]

for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')

def LGB_test(train_x,train_y,test_x,test_y):
    from multiprocessing import cpu_count
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50,random_state=2018,n_jobs=cpu_count()-1
    )
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)
    # print(clf.feature_importances_)
    return clf,clf.best_score_[ 'valid_1']['auc']

def LGB_predict(train_x,train_y,test_x,res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=140, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=4000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.03, min_child_weight=50, random_state=2018, n_jobs=100
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('submission.csv', index=False)
    os.system('zip adFeat_tunePara.zip submission.csv')
    return clf

model=LGB_predict(train_x,train_y,test_x,res)
