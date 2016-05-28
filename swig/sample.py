#!-*- coding:utf8 -*-
import sys

#from sklearn.metrics import accuracy_score

import onlineml

def accuracy_score(x, y):
    n = float(len(x))
    m = 0.
    for i in range(n):
        if x[i]==y[i]:
            m += 1.
    return m/n

p = onlineml.Perceptron()

i = 0
for line in open(sys.argv[1]):
    sp = line.strip().split()
    i += 1
    y_i = sp[0]
    x_i = {}
    for elem in sp[1:]:
        sp_ = elem.split(":")
        ft = sp_[0]
        val = float(sp_[1])
        x_i[ft] = val

    if i % 1000 == 0:
        print(i)
#        break
    p.fit(onlineml.fvs([x_i]), onlineml.labels([y_i]))

p.save("test")

cls = onlineml.Classifier()
cls.load("test")
#sys.exit()

print("test")
y_pred1 = []
y_pred2 = []
y_true = []
for line in open(sys.argv[2]):
    sp = line.strip().split()
    y_true_i = sp[0]
    x_i = {}
    for elem in sp[1:]:
        sp_ = elem.split(":")
        ft = sp_[0]
        val = float(sp_[1])
        x_i[ft] = val

#    print(x_i)
    x_test = onlineml.fv(x_i)

    i1 = p.predict(x_test)
    i2 = cls.predict(x_test)
#    print(i)
    y_pred_i1 = p.id2label(i1)
    y_pred_i2 = cls.id2label(i2)
#    print(y_true_i==y_pred_i, y_true_i, y_pred_i, accuracy_score(y_true, y_pred))

    y_true.append(y_true_i)
    y_pred1.append(y_pred_i1)
    y_pred2.append(y_pred_i2)

    print(y_true_i, y_pred_i1, accuracy_score(y_true, y_pred1))
    print(y_true_i, y_pred_i2, accuracy_score(y_true, y_pred2))
    print

print accuracy_score(y_true, y_pred1)
print accuracy_score(y_true, y_pred22)
