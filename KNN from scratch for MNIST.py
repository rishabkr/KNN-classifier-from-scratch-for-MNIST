import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from collections import Counter

class KNNClassifier:
    def __init__(self):
        pass
    def train_test_split(self, dataframe,test_size):
        dataframe_size=len(dataframe)
        if isinstance(test_size,float):#if test size is passed as a proportion
            test_size=round(test_size*dataframe_size)
        #pick random samples from the data for train test split
        indexes=dataframe.index.tolist()
        test_indices=random.sample(population=indexes,k=test_size)
        #now putting the values of train and test data into the respective df's
        test_dataframe=dataframe.loc[test_indices]
        cropped_dataframe=dataframe.drop(test_indices)
        train_dataframe=cropped_dataframe
        return train_dataframe,test_dataframe
    
    def minkowski(self,test_value,p):
        if(p==2):
            distance=np.sum((self.train_values - test_value)**2,axis=1)
            return distance
        elif(p==1):
            distance=np.sum(abs(self.train_values - test_value),axis=1)
            return distance


    def KNeighbors(self, k, test_value,p=2):
        neighbors=[]
        train_length=self.train_values.shape[0]
        if(p==2):
            distance=self.minkowski(test_value,p=2)
        elif(p==1):
            distance=self.minkowski(test_value,p=1)
        k_neighbors=np.argsort(distance)
        k_neighbors=k_neighbors[:k]
        return k_neighbors

    def find_majority(self, k_index):
        ans = Counter(self.train_labels[k_index]).most_common()
        return ans[0][0]

    def train(self, train_path):
        df=pd.read_csv(train_path)
        train_df,test_df=self.train_test_split(df,0.3)
        test_df,val_df=self.train_test_split(test_df,0.5)

        train_digits=train_df.to_numpy()
        train_digits=np.array(train_digits)
        val_digits=val_df.to_numpy()
        val_digits=np.array(val_digits)
        test_digits=test_df.to_numpy()
        test_digits=np.array(test_digits)

        self.train_values=train_digits[:,1:]
        self.train_labels=train_digits[:,0]
        test_values=test_digits[:,1:]
        test_labels=test_digits[:,0]
        val_values=val_digits[:,1:]
        val_labels=val_digits[:,0]

    def predict(self, test_path):
        df_test=pd.read_csv(test_path,header=None)
        test_vals=df_test.to_numpy()
        test_vals=np.array(test_vals)
        predictions=[]
        length=test_vals.shape[0]
        for i in range(length):
            k_index=self.KNeighbors(5,test_vals[i])
            result=self.find_majority(k_index)
            predictions.append(result)
        return np.array(predictions)


if __name__ == '__main__':
    knn = KNNClassifier()
    knn.train("train.csv")
    preds = knn.predict("test.csv")
    print("Done Testing")
    df_labels=pd.read_csv("test_labels.csv")
    label_vals=df_labels.iloc[:, 0].to_numpy()
    label_vals=np.array(label_vals)
    print(preds.shape)
    print(label_vals.shape)
    acc = np.sum(preds == label_vals)/preds.shape[0]
    print(acc)
'''
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(label_vals, predictions)
print(cm)

cnt=0
for i in range(0,length):
    if(predictions[i]==label_vals[i,0]):
        cnt+=1
print(cnt)

train_labels[25]

def drawing(sample):
    img = sample.reshape((28,28))
    plt.imshow(img,cmap = 'gray')
    plt.show()

drawing(test_values[1])
print(test_labels[1])



from sklearn.neighbors import KNeighborsClassifier





classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)





classifier.fit(train_values,train_labels)



y_pred=classifier.predict(test_vals)




from sklearn.metrics import confusion_matrix





cm=confusion_matrix(y_pred,label_vals)





'''