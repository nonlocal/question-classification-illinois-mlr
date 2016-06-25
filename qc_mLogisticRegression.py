file = open('qc_data.txt','r')
qc_dataset = file.read()
qc_dataset = qc_dataset.split('\n')
del qc_dataset[-1]
index = len(qc_dataset)
labels = []
features = []
for i in range(index):
    labels.append(qc_dataset[i].split()[0])
for i in range(index):
    features.append(" ".join( qc_dataset[i].split()[1:] ))
from sklearn.cross_validation import train_test_split
train_feat, test_feat, train_label, test_label = train_test_split(features,labels,test_size=0.15)
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
train_vect = count_vect.fit_transform(train_feat)
test_vect = count_vect.transform(test_feat)
##################################################
#       LogisticRegression Multinomial
##################################################
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(multi_class='multinomial',solver='lbfgs')
clf_lr = clf_lr.fit(train_vect, train_label)
pr_lr = clf_lr.predict(test_vect)
print(clf_lr.score(test_vect, test_label))
#sample score: 0.959447799827
close(file)
