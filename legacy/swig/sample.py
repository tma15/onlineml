#!-*- coding:utf8 -*-
import sys

import onlineml


#p = onlineml.Perceptron()
p = onlineml.AveragedPerceptron()

i = 0
x_train = onlineml.PairVectors()
y_train = onlineml.StringVectors()

for line in open(sys.argv[1]):
    sp = line.strip().split()
    i += 1
    y_i = sp[0]
    x_i = onlineml.PairVector()
    for elem in sp[1:]:
        sp_ = elem.split(":")
        ft = sp_[0]
        val = float(sp_[1])
        x_i.append((ft, val))

    x_train.append(x_i)
    y_train.append(y_i)
    if i % 10000 == 0:
        print("reading training data: {}".format(i))
#        break

print("training model ...")
epoch = 3
for i in range(epoch):
    p.fit(x_train, y_train)

print("saving model ...")
p.save("model-swig-python")

print("loading model ...")
cls = onlineml.Classifier()
cls.load("model-swig-python")

accuracy_score = 0
num_corr = 0
num_total = 0
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

    pred_i = cls.predict(x_test)
    y_pred_i = cls.id2label(pred_i)

    if y_true_i == y_pred_i:
        num_corr += 1
    num_total += 1

    if num_total % 1000 == 0:
        accuracy_score = float(num_corr) / float(num_total)
        print("Acc:{0:.3f} ({1}/{2}) i:{3} true:{4} pred:{5}".format(accuracy_score,
            num_corr, num_total, num_total, y_true_i, y_pred_i))

accuracy_score = float(num_corr) / float(num_total)
print(accuracy_score)
