from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC,SVC


from sklearn.metrics import classification_report

# 1.获取数据
news = fetch_20newsgroups(subset='all')
# print(len(news.data))  # 输出数据的条数：18846
# print(news.data[0])

# 2.数据预处理
# 分割
xTrain,xTest,yTrain,yTest = train_test_split(news.data,news.target,test_size=0.25,random_state=1) # 随机采样25%的数据样本作为测试集
# print("原数据:\n",xTrain[0])  #查看训练样本
# print(yTrain[0:100])  #查看标签
# 文本向量化
vec = CountVectorizer()
xTrain = vec.fit_transform(xTrain) #fit_transform是fit和transform的结合
xTest = vec.transform(xTest)
# print("特征化后的数据:\n",xTrain[0])
# print(xTest)
# xTrain = xTrain.astype('float32')
# xTest = xTest.astype('float32')
# yTrain = yTrain.astype('float32')
# yTest = yTest.astype('float32')

# 3.1 朴素贝叶斯
def MNB():
    mnb = MultinomialNB()  # 使用默认配置初始化朴素贝叶斯
    mnb.fit(xTrain, yTrain)  # 利用训练数据对模型参数进行估计
    mnb_predict = mnb.predict(xTest)  # 对参数进行预测
    print('朴素贝叶斯分类的准确率为:', mnb.score(xTest, yTest))
    return mnb_predict
    # for i in range(1,20):
    #     mnb = MultinomialNB(alpha=i/10)   # 使用默认配置初始化朴素贝叶斯
    #     mnb.fit(xTrain,yTrain)    # 利用训练数据对模型参数进行估计
    #     mnb_predict = mnb.predict(xTest)     # 对参数进行预测
    #     print("alpha={}时,朴素贝叶斯准确率为{}".format(i/10,mnb.score(xTest,yTest)))

#3.2 高斯贝叶斯
def GNB():
    gnb = GaussianNB()
    gnb.fit(xTrain,yTrain)
    gnb_predict = gnb.predict(xTest)
    print('高斯贝叶斯分类的准确率为:', gnb.score(xTest, yTest))
    return gnb_predict
#3.3 伯努利贝叶斯
def BNB():
    bnb = BernoulliNB(alpha=0.0000001)
    bnb.fit(xTrain, yTrain)
    bnb_predict = bnb.predict(xTest)
    print('伯努利贝叶斯分类的准确率为:', bnb.score(xTest, yTest))
    return bnb_predict
    # for i in range(1,20):
    #         bnb = BernoulliNB(alpha=i/10)
    #         bnb.fit(xTrain,yTrain)
    #         bnb_predict = bnb.predict(xTest)
    #         print("alpha={}时,伯努利贝叶斯准确率为{}".format(i/10,bnb.score(xTest,yTest)))

#3.4 岭回归分类
def RC():
    # rc = RidgeClassifier()
    # rc.fit(xTrain,yTrain)
    # rc_predict = rc.predict(xTest)
    # print('岭回归分类的准确率为:', rc.score(xTest, yTest))
    # return rc_predict
    import datetime
    starttime = datetime.datetime.now()
    for i in [0.2,0.4,0.6,0.8]:
        rc = RidgeClassifier(alpha = i)
        rc.fit(xTrain,yTrain)
        rc_predict = rc.predict(xTest)
        print("alpha={}时,岭回归分类准确率为{}".format(i, rc.score(xTest, yTest)))
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
# 3.5 逻辑回归分类
def LR():
    # lr = LogisticRegression(C=0.05, multi_class="multinomial", solver="newton-cg")
    # lr.fit(xTrain, yTrain)
    # lr_predict = lr.predict(xTest)
    # print('逻辑回归分类的准确率为:', lr.score(xTest, yTest))
    # return lr_predict
    import datetime
    starttime = datetime.datetime.now()
    for C in range(1,20):
        lr = LogisticRegression(C=C,multi_class="multinomial",solver="newton-cg")
        lr.fit(xTrain,yTrain)
        lr_predict = lr.predict(xTest)
        print("C={}时，逻辑回归分类准确率为{}".format(C/10 , lr.score(xTest,yTest)))
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
# 3.6 多层感知机
def MLP():
    for i in range(10,80,10):
        for j in range(10,80,10):
            mlp= MLPClassifier(hidden_layer_sizes=(i,j), random_state=1)
            mlp.fit(xTrain,yTrain)
            mlp_predict=mlp.predict(xTest)
            print('当神经元个数为({},{})时，多层感知机分类的准确率为:{}'.format(i,j,mlp.score(xTest, yTest)))

# 3.7 lightBLM
def GBM():
    gbm = LGBMClassifier(num_leaves=31,learning_rate=0.05,n_estimators=10)
    gbm.fit(xTrain, yTrain,eval_set=[(xTest, yTest)],early_stopping_rounds=5)
    gbm_predict = gbm.predict(xTest, num_iteration=gbm.best_iteration_)
    print('lightBLM分类的准确率为:', gbm.score(xTest,yTest))
    return gbm_predict
# 3.8 随机森林
def RFC():
    rfc = RandomForestClassifier(random_state=1)
    rfc.fit(xTrain, yTrain)
    rfc_predict = rfc.predict(xTest)
    print('随机森林分类的准确率为:', rfc.score(xTest, yTest))
    return rfc_predict

# 获取结果报告
# print(classification_report(yTest, MLP(), target_names = news.target_names))