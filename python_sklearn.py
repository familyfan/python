# -*- coding: utf-8 -*-
from sklearn.metrics import  confusion_matrix, accuracy_score,f1_score,roc_auc_score,recall_score,precision_score


confusion_matrix(y_true, y_pred)
pred = multilayer_perceptron(x, weights, biases)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
init = tf.initialize_all_variables()
sess.run(init)
for epoch in xrange(150):
    for i in xrange(total_batch):
        train_step.run(feed_dict = {x: train_arrays, y: train_labels})
        avg_cost += sess.run(cost, feed_dict={x: train_arrays, y: train_labels})/total_batch         
    if epoch % display_step == 0:
        print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

#metrics
y_p = tf.argmax(pred, 1)
val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_arrays, y:test_label})

print("validation accuracy:", val_accuracy)
y_true = np.argmax(test_label,1)
print("Precision", sk.metrics.precision_score(y_true, y_pred))
print( "Recall", sk.metrics.recall_score(y_true, y_pred))
print( "f1_score", sk.metrics.f1_score(y_true, y_pred))
print( "confusion_matrix")
print( sk.metrics.confusion_matrix(y_true, y_pred))
fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)



#例如有混淆矩阵
#样本标签顺序:'0','1';分别表示第一行为'0'标签类别，第二行为'1'类别标签
# [10 0]
# [5  5]

#计算准确率
acc_score = accuracy_score(y_test,preds)
print(acc_score)
#0.75 ,计算过程(10+5)/(10+0+5+5)=0.75
#其实也相当于np.mean(preds == y_test))

#计算roc面积，但只限制于2分类，根据preds值的类型做不同处理
#1.preds是预测的类别，而不是概率
#根据上述例子，preds 里面的值是一个类别标签，比如'0'或者'1'
roc_auc_score1 = roc_auc_score(y_test,preds)
print(roc_auc_score1)
#0.75 计算面积如下图(文章最后)，点(0.5,1.0)，分别和(0.0)和(1.1)相连，然后求面积
#2.preds是概率值，那么sklearn会自动调整阈值然后画出roc曲线，然后计算roc曲线面积AUC

#计算召回率
#def recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary',sample_weight=None)
#有个参数pos_label，指定计算的类别的召回率，默认为1，这里也就是查看标签为'1'的召回率
raca_score = recall_score(y_test,preds)
print(raca_score)
#0.5 计算过程5/(5+5)=0.5

#计算正确率
#def precision_score(y_true, y_pred, labels=None, pos_label=1,average='binary', sample_weight=None)
#有个参数pos_label，指定计算的类别的正确率，默认为1
prec_score = precision_score(y_test,preds)
print(prec_score)
#1.0 计算过程5/(5+0) = 1.0

#计算F1score
#def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary',sample_weight=None)
#使用的是调和平均 harmonic mean,F1 = 2 * (precision * recall) / (precision + recall)
f1_score = f1_score(y_test,preds)
#0.666666666667 计算过程 2(1*0.5)/(1+0.5) = 1/1.5 = 0.666666666667





























#例如有混淆矩阵
#样本标签顺序:'0','1';分别表示第一行为'0'标签类别，第二行为'1'类别标签
# [10 0]
# [5  5]

#计算准确率
acc_score = accuracy_score(y_test,preds)
print(acc_score)
#0.75 ,计算过程(10+5)/(10+0+5+5)=0.75
#其实也相当于np.mean(preds == y_test))

#计算roc面积，但只限制于2分类，根据preds值的类型做不同处理
#1.preds是预测的类别，而不是概率
#根据上述例子，preds 里面的值是一个类别标签，比如'0'或者'1'
roc_auc_score1 = roc_auc_score(y_test,preds)
print(roc_auc_score1)
#0.75 计算面积如下图(文章最后)，点(0.5,1.0)，分别和(0.0)和(1.1)相连，然后求面积
#2.preds是概率值，那么sklearn会自动调整阈值然后画出roc曲线，然后计算roc曲线面积AUC

#计算召回率
#def recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary',sample_weight=None)
#有个参数pos_label，指定计算的类别的召回率，默认为1，这里也就是查看标签为'1'的召回率
raca_score = recall_score(y_test,preds)
print(raca_score)
#0.5 计算过程5/(5+5)=0.5

#计算正确率
#def precision_score(y_true, y_pred, labels=None, pos_label=1,average='binary', sample_weight=None)
#有个参数pos_label，指定计算的类别的正确率，默认为1
prec_score = precision_score(y_test,preds)
print(prec_score)
#1.0 计算过程5/(5+0) = 1.0

#计算F1score
#def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary',sample_weight=None)
#使用的是调和平均 harmonic mean,F1 = 2 * (precision * recall) / (precision + recall)
f1_score = f1_score(y_test,preds)
#0.666666666667 计算过程 2(1*0.5)/(1+0.5) = 1/1.5 = 0.666666666667

