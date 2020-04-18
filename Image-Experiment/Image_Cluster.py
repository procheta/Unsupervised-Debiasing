#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv
import numpy as np
import math
import sklearn
import tensorflow.compat.v1 as tf
import sys

#Parameters Declaration
tf.disable_v2_behavior()
nCluster=4
biasLabel=[]
genderMap = dict({"1":"Male", "-1":"Female"})
attractionMap= dict({"0":"Not-Attractive", "1":"Attractive"})
ageMap=dict({"1":"young", "-1":"Old"})
raceMap= dict({"0":"African-American", "1":"European", "2":"NA"})
OnehotVecPath='C:/users/Procheta/Downloads/celebs.20k.vae.vec/celebs.20k.vec'
#OnehotVecPath='C:/users/Procheta/Downloads/celeba.784.mobnet.vecs'


def loadVaeVec(OnehotVecPath):
    imageIds=[]
    passages=[]
    with open(OnehotVecPath) as csv_file:
        for idx in csv_file:
            line= idx.split("\t")            
            imageIds.append(line[0])
            tokens = line[1].split(' ')
            vec=[]
            for token in tokens:
                try:
                    vec.append(float(token))
                except:
                    d=0  
            passages.append(vec)
    return passages, imageIds

def loadMobileNetVec(OnehotVecPath, imageIdPath):
    imageIds=[]
    with open(imageIdPath) as csv_file:
        for idx in csv_file:
            line= idx.split("\n")            
            imageIds.append(line[0])     
    
    passages=[]
    
    with open(OnehotVecPath) as csv_file:
        for idx in csv_file:
            line= idx.split(" ")                        
            vec=[]
            for token in line:
                try:
                    vec.append(float(token))
                except:
                    d=0  
            passages.append(vec)
    return passages, imageIds

def loadAnAttribute(filePath, index):
    count1=0
    labels=[]
    with open(filePath) as csv_file:
        for idx in csv_file:
            line=idx.split("\n")[0];
            tokens = line.split(',')
            count = 0
            if count1 == 1:    
                for token in tokens:
                    if count==index:
                        labels.append(token)
                    count=count+1
            else:
                count1=1
    return labels

def computeProbability(predicted_classes,genderLabel,gender, emotion,startIndex, ageLabel, age):
    nominator=0
    denominator=0
    for i in range(len(predicted_classes)):
        if (genderLabel[i]==str(gender) and predicted_classes[i]==emotion and ageLabel[i]==str(age) ):
            nominator=nominator+1
        if(predicted_classes[i]==emotion):
            denominator=denominator + 1
    try:     
        prob=nominator/(denominator)
    except:
        prob =0
        print(nominator)
    #if indicator == 1:
        #print("Given the emotion is", emotionMap[str(emotion)], "probability of being", raceMap[str(gender)],prob)
    #else:
    print("Given the persion is", attractionMap[str(emotion)], "probability of being", genderMap[str(gender)], "and", ageMap[str(age)], prob)
    return prob
def computeposterior(input_label):    
    count=[]
    probs=[]
    for i in range(4):
        count.append(0)
        probs.append(0)
        
    for i in range(len(input_label)):
        count[input_label[i]]=count[input_label[i]]+1
        
    probs[0]=count[0]/len(input_label)
    probs[1]=count[1]/len(input_label)
    probs[2]=count[2]/len(input_label)
    probs[3]=count[3]/len(input_label)
   
    
    for i in range(len(input_label)):
        for j in range(len(probs)):
            if input_label[i] == j:
                if probs[j] == .25:
                    biasLabel.append(0)
                else:
                    biasLabel.append(1)
    
    print("Probability of class 0 w.r.t input label 0",count[0]/len(input_label))
    
    print("Probability of class 1 w.r.t input label 0",count[1]/len(input_label))
    
def createTrainOutput(splitIndex, numClasses,OutputLabel):
    output=[]
    for i in range(splitIndex):
        x=np.zeros(numClasses)
        x[OutputLabel[i]]=1
        output.append(x)
    return output 

def estimateLabels(modelOutput):
    probs=[]
    labels=[]
    for i in range(modelOutput.shape[0]):
        index=np.where(modelOutput[i] == np.amax(modelOutput[i]))
        labels.append(index[0][0])
    return labels
        
passages, imageIds=loadVaeVec(OnehotVecPath)
#passages, imageIds=loadMobileNetVec(OnehotVecPath, "C:/Users/Procheta/Downloads/subsampled_data.20000.tar/subsampled_data.20000/imageIds.txt")
print(len(passages))
print(len(passages[0]))
OutputLabel=[]
dict_label={}
OnehotVec='C:/Users/Procheta/Downloads/subsampled_data.20000.tar/subsampled_data.20000/list_attr_celeba.csv'
count1=0
with open(OnehotVec) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx in csv_file:
        tokens = idx.split(',')
        count = 0
        value=""
        key=""
        if count1 == 1:    
            for token in tokens:
                if count == 0:
                    key=token
                    
                if count==3:
                    value=token
                count=count+1
            dict_label[key]=value
        else:
            count1=1
count = 0;
OutputLabel=[]
with open("C:/Users/Procheta/Downloads/attractions.txt","w") as f:
    for i in range(len(imageIds)):
        val = dict_label[imageIds[i]]
        if val == "-1":
            val="0"
            count = count +1
        OutputLabel.append(int(val))
        f.write(str(val))
        f.write("\n")
f.close()
print(len(OutputLabel))
print('Data Loaded')
print('Starting Logistic Regression Model')

length =(int) (len(passages)*.80)
totalLength= len(passages)
model = LogisticRegression(solver='lbfgs', penalty="none")
model.fit(passages[0:length], OutputLabel[0:length])
predicted_classes = model.predict(passages[length:totalLength])

accuracy = accuracy_score(OutputLabel[length:totalLength],predicted_classes)
print('accuracy of the model before de-biasing', accuracy)

genderLabel=loadAnAttribute("C:/Users/Procheta/Downloads/subsampled_data.20000.tar/subsampled_data.20000/list_attr_celeba.csv",21)

ageLabel=loadAnAttribute("C:/Users/Procheta/Downloads/subsampled_data.20000.tar/subsampled_data.20000/list_attr_celeba.csv",40)

prob_0=computeProbability(predicted_classes,genderLabel[length:totalLength],1, 1,length, ageLabel,1)
prob_1=computeProbability(predicted_classes,genderLabel[length:totalLength],1, 1,length, ageLabel,-1)

prob_4=computeProbability(predicted_classes,genderLabel[length:totalLength],-1, 1,length, ageLabel,1)
prob_5=computeProbability(predicted_classes,genderLabel[length:totalLength],-1, 1,length, ageLabel,-1)


prob_2=computeProbability(predicted_classes,genderLabel[length:totalLength],-1, 0,length, ageLabel,1)
prob_3=computeProbability(predicted_classes,genderLabel[length:totalLength],-1, 0,length, ageLabel,-1)

prob_2=computeProbability(predicted_classes,genderLabel[length:totalLength],1, 0,length, ageLabel,1)
prob_3=computeProbability(predicted_classes,genderLabel[length:totalLength],1, 0,length, ageLabel,-1)

fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing',attractionMap[str(1)],'is',gamma)

input_data=passages[0:length]
input_label=OutputLabel[0:length]
group_cluster=[]

for i in range(2):
    x=[]
    group_cluster.append(x)


for i in range(len(input_label)):
    group_cluster[input_label[i]].append(passages[i])
    
print("First level grouping done...")

for i in range(len(group_cluster)):
    group = group_cluster[i]
    kmeans = KMeans(n_clusters=4, random_state=0).fit(group)
    computeposterior(kmeans.labels_)
    print("Clustering complete for Attraction level",attractionMap[str(i)])

input_label=createTrainOutput(length, 2,OutputLabel)  

def multiTaskModelWithBias(train_input,train_output,bias_output,gamma,test_input,splitIndex,numClasses,midDim):
    X = tf.placeholder("float", [splitIndex,len(train_input[0])], name="X")
    Y1 = tf.placeholder("float",[splitIndex,2], name="Y1")
    Y2 = tf.placeholder("float", [splitIndex,1], name="Y2")

    tf.set_random_seed(1236)
    np.random.seed(1234)

    initial_shared_layer_weights = np.random.rand(len(train_input[0]),midDim)
    initial_Y1_layer_weights = np.random.rand(midDim, numClasses)
    initial_Y2_layer_weights = np.random.rand(midDim,1)

    shared_layer_weights = tf.Variable(initial_shared_layer_weights, name="share_W", dtype="float32")
    Y1_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y1", dtype="float32")
    Y2_layer_weights = tf.Variable(initial_Y2_layer_weights, name="share_Y2", dtype="float32")

    shared_layer = tf.nn.relu(tf.matmul(X,shared_layer_weights))
    Y1_layer = tf.nn.sigmoid(tf.matmul(shared_layer,Y1_layer_weights))

    #constant_matrix=tf.cast(constant_matrix, tf.float32)
    #subspace_layer=tf.multiply(shared_layer,constant_matrix)
    Y2_layer = tf.nn.sigmoid(tf.matmul(shared_layer,Y2_layer_weights))

    # Calculate Loss
    Y1_Loss = tf.nn.l2_loss(Y1-Y1_layer)
    Y2_Loss = tf.nn.l2_loss(Y2-Y2_layer)
    Y2_Loss = -Y2_Loss
    #Joint_Loss = gamma*Y1_Loss+(1-gamma)*Y2_Loss

    #Optimiser = tf.train.AdamOptimizer().minimize(Joint_Loss)
    Y1_op = tf.train.AdamOptimizer().minimize(Y1_Loss)
    Y2_op = tf.train.AdamOptimizer().minimize(Y2_Loss)
    #Y1_op =tf.train.GradientDescentOptimizer(learning_rate = .1).minimize(Y1_Loss)
    Y1_layer_wts=0
    shared_layer_wts=0
    Y1_layer_val =0
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        for i in range(2000):
           _,tt,Y1_loss_val, Y2_Loss_val, shared_layer_wts, Y1_layer_wts,Y2_layer_wts = session.run([Y1_op,Y2_op,Y1_Loss, Y2_Loss, shared_layer_weights, Y1_layer_weights, Y2_layer_weights],{X: train_input,Y1: train_output,Y2: bias_output})
           if i%10 == 0:
               print(Y1_loss_val," ",Y2_Loss_val)
        test_input= tf.cast(test_input, tf.float32)
        shared_layer = tf.nn.relu(tf.matmul(test_input,shared_layer_wts))
        Y1_layer_val = tf.nn.sigmoid(tf.matmul(shared_layer,Y1_layer_wts))
        Y1=session.run(Y1_layer_val)

    return Y1


biasLabel=np.asarray(biasLabel)
biasLabel=biasLabel.reshape(length,1)
Y2=multiTaskModelWithBias(input_data, input_label, biasLabel, .2,passages[length:totalLength],length,2,200)
print('bias aware model training complete')

labels=estimateLabels(Y2)
accuracy = accuracy_score(OutputLabel[length:totalLength],labels)
print('Accuracy of the model after de-biasing', accuracy)


#prob_0=computeProbability(labels,genderLabel,1, 1,length)
#prob_1=computeProbability(labels,genderLabel,-1, 1,length)
#fairness_score=float(prob_0)*float(prob_1)
#print('Fairness after debiasing approach',fairness_score)
#gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
#print('Gamma value after debiasing',gamma)


prob_0=computeProbability(labels,genderLabel[length:totalLength],1, 1,length, ageLabel,1)
prob_1=computeProbability(labels,genderLabel[length:totalLength],1, 1,length, ageLabel,-1)

prob_4=computeProbability(labels,genderLabel[length:totalLength],-1, 1,length, ageLabel,1)
prob_5=computeProbability(labels,genderLabel[length:totalLength],-1, 1,length, ageLabel,-1)


prob_2=computeProbability(labels,genderLabel[length:totalLength],-1, 0,length, ageLabel,1)
prob_3=computeProbability(labels,genderLabel[length:totalLength],-1, 0,length, ageLabel,-1)

prob_2=computeProbability(labels,genderLabel[length:totalLength],1, 0,length, ageLabel,1)
prob_3=computeProbability(labels,genderLabel[length:totalLength],1, 0,length, ageLabel,-1)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing for emotion',emotionMap[str(0)],'is',gamma)

prob_0=computeProbability(labels,genderLabel[length:totalLength],genderLabel,1, 1,length,1)
prob_1=computeProbability(labels,genderLabel[length:totalLength],genderLabel,0, 1,length,1)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing',emotionMap[str(1)],'is',gamma)

prob_0=computeProbability(labels,OutputLabel,genderLabel,1, 3,length,1)
prob_1=computeProbability(labels,OutputLabel,genderLabel,0, 3,length,1)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing',emotionMap[str(3)],'is',gamma)


# In[ ]:








