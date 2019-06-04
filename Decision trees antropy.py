import pandas as pd
import numpy as np


def raw_data_train_test(data,test_split):
    a=pd.read_csv(data,header=None)
#    a.iloc[:,1] = a.iloc[:,1].apply(lambda x: reutrn 0 if x=='M'  else return 1)
    for i in range(len(a)):
        if a.iloc[i,1]=="M":
            a.iloc[i,1]=0
        else:
            a.iloc[i,1]=1
    a=a.values
    c=a[:,1]
    a=np.delete(a, np.s_[1], 1)
    a=np.column_stack([a,c])
    length=a.shape[0]
    d_train=a[0:(int(length*(1-0.2)))]
    d_test=a[length-(int(length*0.2)):length]
    return d_train, d_test
d_train, d_test =raw_data_train_test( r"http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",0.1)

def split_data(data,col_index,val):
    d_L=np.array([])
    d_R=np.array([])
    for i in range(len(data)):
        if val>=data[i,col_index]:
            d_L=np.insert(d_L, 0, data[i], axis=0)
        else:
            d_R=np.insert(d_R, 0, data[i], axis=0)
    d_L=d_L.reshape(int(len(d_L)/data.shape[1]),data.shape[1])
    d_R=d_R.reshape(int(len(d_R)/data.shape[1]),data.shape[1])
    return d_L, d_R

def entropy(data):
    temp1=0
    temp2=0
    if  len(data)==0 :
        entropy=1
    else:
        for i in range(len(data)):
             if data[i,data.shape[1]-1]==1:
                 temp1=temp1+1
             else:
                 temp2=temp2+1
             entropy  = (-1*(temp1/len(data))*(np.log((temp1/len(data))+(np.finfo(float).eps))))\
             -((temp2/len(data))*(np.log((temp2/len(data))+(np.finfo(float).eps))))
    return entropy

def get_split(data):
    best_gain=0
    col_index=0
    val=0
    first_entropy=entropy(data)
    for x in range(data.shape[1]-1):
        for i in range(len(data)):
                d_L, d_R=split_data(data,x,data[i,x])
                d_L_entropy=entropy(d_L)
                d_R_entropy=entropy(d_R)
                avg=(d_L_entropy*(len(d_L)/len(data)))+(d_R_entropy*(len(d_R)/len(data)))
                gain=first_entropy-avg
                if best_gain<gain:
                    best_gain=gain
                    col_index=x
                    val=data[i,x]
    return col_index, val 

def leaf(data):
    a=True
    for i in range(len(data)-1):
        if data[i,data.shape[1]-1]!=data[i+1,data.shape[1]-1]:
            a=False
            break
    return a

def lable_check(data):
    a=data[:,data.shape[1]-1].mean()
    if a>=0.5:
        return 1
    else:
        return 0
   
class Node:
    def __init__(self,data,max_depth):
        self.data=data
        self.max_depth=max_depth
        self.left_node=None
        self.right_node = None
        self.lable=None
        
    def build_tree(self):   
       if self.max_depth == 0 or leaf(self.data):
           self.lable = lable_check(self.data)
           return  
                     
       self.col_index,self.val=get_split(self.data) 
       self.left_data,self.right_data=split_data(self.data,self.col_index,self.val)  
       
       self.left_node = Node(self.left_data,self.max_depth-1)
       self.left_node.build_tree()
       self.right_node = Node(self.right_data,self.max_depth-1)
       self.right_node.build_tree()
       
    def predict(self,test_row):
       if self.lable is not None:
           return self.lable
       if test_row[:,self.col_index]<=self.val:
           return self.left_node.predict(test_row)         
       if test_row[:,self.col_index]>self.val:
           return self.right_node.predict(test_row)           
           
        

root=Node(d_train,8)  
root.build_tree()
def Accuracy(data):
    correct = 0
    for i in range(len(data)):
        temp=root.predict(data[i,:].reshape(1,32))
        if temp == data[i,data.shape[1]-1]:
            correct += 1	
        Accuracy=((correct/(len(data))) * 100.0)
    return Accuracy
print(Accuracy(d_test))
