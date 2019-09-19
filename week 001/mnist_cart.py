from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets import load_digits

#加载数据
digits = load_digits()
data = digits.data

# 分割数据
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25,random_state=33)

ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

#
clf = tree.DecisionTreeRegressor()
clf.fit(train_ss_x, train_y)

predict = clf.predict(test_ss_x)

print('CART 准确率：%0.41f' % accuracy_score(predict, test_y))