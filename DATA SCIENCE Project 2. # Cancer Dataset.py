import pandas as pd
df = pd.read_csv('cancer_dataset.csv',header = None)
print(df.shape)
print(df.head())


df.describe()


# In[76]:


df.info()


# In[77]:


df[1].value_counts()


# In[78]:


import seaborn as sns
sns.countplot(df[1])


# In[79]:


sns.distplot(df[1])


# In[107]:


corr = df.corr()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
plt.show()


# In[81]:


df = df.drop(0,axis = 1)
print(df)


# In[124]:


df.isna().sum()


# In[82]:


dict = {'B':0,'M':1}
df1 = df
df1[1]= df1[1].map(dict)
print(df1)


# In[83]:


df1[1].value_counts()


# In[84]:


x = df1.iloc[:,2:11]
y = df1[1]


# In[85]:


print(x)
print(y)


# In[86]:


print((y==0).sum())


# In[87]:


import numpy as np
print(np.unique(y))


# In[88]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test =train_test_split(x,y,test_size =.20)
print(x.shape)
print(x_train.shape)
print(x_test.shape)


# In[89]:


from sklearn.linear_model import Perceptron
clf = Perceptron()
clf.fit(x_train,y_train)
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)


# In[90]:


from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print('Train_acc=',train_acc*100)
print('test Accuracy=',test_acc*100)


# In[91]:


from sklearn.metrics import classification_report
cr = classification_report(y_test,y_test_pred)
print(cr)


# In[92]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_test_pred)
print(cm)


# In[93]:


import seaborn as sns
sns.heatmap(cm,annot =True,cbar = True)


# In[94]:


#Model 2 LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train,y_train)
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train,y_train_pred)
test_acc = accuracy_score(y_test,y_test_pred)
print('Training Acc=', train_acc*100)
print('Testing Acc=',test_acc*100)


# In[95]:


#Model 3 SVC Support VectorMAchine
from sklearn.svm import SVC
clf = SVC()
clf.fit(x_train,y_train)
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train,y_train_pred)
test_acc = accuracy_score(y_test,y_test_pred)
print('Training Acc=', train_acc*100)
print('Testing Acc=',test_acc*100)


# In[96]:


#Model 4 KNeighborsClassifier(KNN)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(x_train,y_train)
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train,y_train_pred)
test_acc = accuracy_score(y_test,y_test_pred)
print('Training Acc=', train_acc*100)
print('Testing Acc=',test_acc*100)


# In[108]:


####Compare the performance of different Classifiers (MOdels)
from sklearn.linear_model import Perceptron,LogisticRegression  #(logic function in logistic regresson f(x) = 1/(1+e^-x))
                                                                #perceptron is a hard limit classifier
from sklearn.svm import SVC        # Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier     # All above models are Basic Classifiers...Algo of DTC - CART, ID3,C4.5
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier # Bagging (dividing in random samples with replacement)
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier   #Boosting  (increase the weight of misclassified samples)
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier  #Stacking  (pass the whole data into every classifier one by one)
clf1 = Perceptron()
clf2 = LogisticRegression()
clf3 = SVC()
clf4 = KNeighborsClassifier()
clf5 = GaussianNB()
clf6 = DecisionTreeClassifier()
clf7 = RandomForestClassifier()
clf8 = BaggingClassifier()
clf9 = ExtraTreesClassifier()
clf10 = AdaBoostClassifier()
clf11= GradientBoostingClassifier()
clf12= VotingClassifier(estimators = [('per',clf1),('dt',clf6),('ada',clf10)],voting='hard')
clf = [clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8,clf9,clf10,clf11,clf12]
name = ['Per','LR','SVC','KNN','GNB','DT','RF','BAG','ET','ADA','GBC','VT']
accuracy = {}
for model,model_name in zip(clf,name):
    model.fit(x_train,y_train)
    acc = accuracy_score(model.predict(x_test),y_test)
    accuracy[model_name] = acc
print(accuracy)




# In[109]:


for i,j in accuracy.items():
    print(i,':',j*100)


# In[125]:


classifiers=[]
ac=[]
for i,j in accuracy.items():
    classifiers.append(i)
    ac.append(j)
fig=plt.figure(figsize=(10,5))
# bar plot 
plt.bar(classifiers,ac,color="Black")
plt.xlabel("models")
plt.ylabel("accuracy")
plt.title("accuracy of different")
plt.show()


# In[129]:


#Hyperparameter Tuning
clf = RandomForestClassifier(n_estimators = 50,min_samples_split=20,max_depth=7,max_features=1)
clf.fit(x_train,y_train)
y_train_pred = clf7.predict(x_train)
y_test_pred = clf7.predict(x_test)
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train,y_train_pred)
test_acc = accuracy_score(y_test,y_test_pred)
print('Training Acc=', train_acc*100)
print('Testing Acc=',test_acc*100)
print('Cross Validation',model.score(x_test,y_test)*100)


# In[130]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=None)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
pipe_svc = Pipeline([('scl', StandardScaler()),
           ('pca', PCA(n_components=3)),
           ('clf', SVC(random_state=1))])


# In[131]:


from sklearn.model_selection import GridSearchCV
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10)
gs = gs.fit(x_train, y_train)
print('Best Training Score ', (gs.best_score_)*100) 
print('Best Parameters value ',gs.best_params_)
clf = gs.best_estimator_
clf.fit(x_train, y_train)
print('Test accuracy: %.3f' % clf.score(x_test, y_test))


# In[ ]:




