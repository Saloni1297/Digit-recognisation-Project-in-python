#Header files
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sc
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pickle as pk

#Initialising list
x_data=[]
y_data=[]

#Reading data
with open('mnist_data') as h:
    x_data=pk.loads(h.read())

with open('mnist_data_label') as ht:
    y_data=pk.loads(ht.read())
    
#Trainning data
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2)

#Converting higher dimension to two-dimension
pca = PCA(n_components=2)
pca.fit(x_train)
new_data=pca.transform(x_train)
m,n=[],[]
for i in new_data:
    m.append(i[0])
    n.append(i[1])
    
#Plotting graph
plt.scatter(m,n,c=y_train)
plt.show()
model=svm.SVC()
model.fit(x_train,y_train)
pred=model.predict(x_test)

#Calculating accuracy
print("Accuracy=")
print accuracy_score(y_test,pred)*100

    
#plt.scatter(x_data,y_data)
#plt.show()
'''x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2)
model=svm.SVC()
model.fit(x_train,y_train)
pred=model.predict(x_test)
l=len(pred)
pred[0:10]
flag=pred==y_test
print accuracy_score(y_test,pred)*100'''
    

