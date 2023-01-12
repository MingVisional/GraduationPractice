import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report
from collections import Counter
import numpy as np
from pylab import *  # 支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

train_df = pd.DataFrame(pd.read_csv('./data/train_set.csv',sep='\t'))
# print(train_df.shape)
# print(train_df.columns) #Index(['label', 'text'], dtype='object')
data = train_df['text']
text = train_df['label']
label_names =['科技',  '股票', '体育',  '娱乐',  '时政',
              '社会',  '教育', '财经','家居',  '游戏',
              '房产',  '时尚',  '彩票',  '星座']

xTrain,xTest,yTrain,yTest = train_test_split(data,text,test_size=0.25,random_state=1) # 随机采样25%的数据样本作为测试集
# print(xTrain)

# 1. Bag of Words
# vectorizer = CountVectorizer()
# xTrain = vectorizer.fit_transform(xTrain)
# xTest = vectorizer.transform(xTest)
# 2. TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
xTrain = tfidf.fit_transform(xTrain)
xTest = tfidf.transform(xTest)

def Draw(pre,label,name):
    t = [0 for i in range(len(label_names))]
    n = [0 for i in range(len(label_names))]
    label = np.array(label)
    for i in range(len(label)):
        n[label[i]] += 1
        if label[i] == pre[i]:
            t[label[i]] += 1
    # for i in range(len(label_names)):
    #     print("{}\t{}".format(label_names[i], t[i] / n[i]))
    x = [i for i in range(len(label_names))]
    y = [t[i] / n[i] for i in range(len(label_names))]
    plt.plot(x, y, marker='o', mfc='w', label=name)

# 1. 朴素贝叶斯
def MNB():
    mnb = MultinomialNB(alpha=0.0000001)  # 使用默认配置初始化朴素贝叶斯
    mnb.fit(xTrain, yTrain)  # 利用训练数据对模型参数进行估计
    mnb_predict = mnb.predict(xTest)  # 对参数进行预测
    print('朴素贝叶斯分类的准确率为:', mnb.score(xTest, yTest))
    Draw(mnb_predict,yTest,"朴素贝叶斯")
    return mnb_predict

# 2. 逻辑回归
def LR():
    lr = LogisticRegression(C=0.05, multi_class="multinomial", solver="newton-cg")
    lr.fit(xTrain, yTrain)
    lr_predict = lr.predict(xTest)
    print('逻辑回归分类的准确率为:', lr.score(xTest, yTest))
    Draw(lr_predict, yTest, "逻辑回归")
    return lr_predict

# 3.岭回归
def RC():
    rc = RidgeClassifier(alpha=10)
    rc.fit(xTrain,yTrain)
    rc_predict = rc.predict(xTest)
    print('岭回归分类的准确率为:', rc.score(xTest, yTest))
    Draw(rc_predict, yTest, "岭回归")
    return rc_predict

# 4.多层感知机
def MLP():
    mlp= MLPClassifier(hidden_layer_sizes=(60,40), random_state=1)
    mlp.fit(xTrain,yTrain)
    mlp_predict=mlp.predict(xTest)
    print('多层感知机分类的准确率为:{}'.format(mlp.score(xTest, yTest)))
    Draw(mlp_predict, yTest, "多层感知机")
    return mlp_predict

allPredict = []
allPredict.append(MLP())
allPredict.append(RC())
allPredict.append(LR())
allPredict.append(MNB())

testLen = len(allPredict[0])

def Select(list):
    count = Counter(list)
    return count.most_common(1)[0][0]

endPredict = []
for i in range(testLen):
    endPredict.append(Select([predict[i] for predict in allPredict]))
# print(endPredict)

def CalAccuracy(pre,label):
    label = np.array(label)
    l = len(pre)
    t = 0
    for i in range(l):
        if pre[i] == label[i]:
            t += 1
    return t/l

print("使用TF-IDF进行预处理后，四种分类方法的集成学习的分类准确率为",CalAccuracy(endPredict,yTest))
print(classification_report(yTest, endPredict, target_names = label_names))

# print(endPredict)
Draw(endPredict, yTest, "集成学习")
plt.legend()

plt.xlabel(u"种类")  # X轴标签
plt.ylabel("准确率")  # Y轴标签
plt.title("TF-IDF不同方法的分类准确率")  # 标题
plt.xticks([i for i in range(len(label_names))], label_names)
plt.show()