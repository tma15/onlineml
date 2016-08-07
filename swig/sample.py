#!-*- coding:utf8 -*-
import sys

import onlineml

def accuracy_score(x, y):
    n = len(x)
    m = 0.
    for i in range(n):
        if x[i]==y[i]:
            m += 1.
    return m/float(n)

p = onlineml.Perceptron()

i = 0
x_train = []
y_train = []
for line in open(sys.argv[1]):
    sp = line.strip().split()
    i += 1
    y_i = sp[0]
    x_i = []
    for elem in sp[1:]:
        sp_ = elem.split(":")
        ft = sp_[0]
        val = float(sp_[1])
        x_i.append((ft, val))

    x_train.append(x_i)
    y_train.append(y_i)
    if i % 1000 == 0:
        print(i)
        break
p.fit2(onlineml.PairVectors(x_train), onlineml.StringVectors(y_train))

p.save2("model-swig-python")

cls = onlineml.Classifier()
cls.load2("model-swig-python")

y_pred1 = []
y_pred2 = []
y_true = []
i = 0
for line in open(sys.argv[2]):
    sp = line.strip().split()
    y_true_i = sp[0]
    x_i = []
    for elem in sp[1:]:
        sp_ = elem.split(":")
        ft = sp_[0]
        val = float(sp_[1])
        x_i.append((ft, val))

    x_test = onlineml.PairVector(x_i)

    i1 = p.predict2(x_test)
    i2 = cls.predict2(x_test)
    y_pred_i1 = p.id2label(i1)
    y_pred_i2 = cls.id2label(i2)

    y_true.append(y_true_i)
    y_pred1.append(y_pred_i1)
    y_pred2.append(y_pred_i2)

    if i % 1000 == 0:
        print(y_true_i, y_pred_i1, accuracy_score(y_true, y_pred1))
        print(y_true_i, y_pred_i2, accuracy_score(y_true, y_pred2))
        print("")
    i += 1

print accuracy_score(y_true, y_pred1)
print accuracy_score(y_true, y_pred22)
