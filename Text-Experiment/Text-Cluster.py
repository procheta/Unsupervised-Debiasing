from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv
import numpy as np
import math
import sklearn
import tensorflow as tf
import sys
#Parameters Declaration
nCluster=5
biasLabel=[]
genderMap = dict({"1":"female", "0":"male"})
emotionMap= dict({"0":"anger", "1":"sadness", "2":"fear", "3":"joy", "4":"no_emotion"})
raceMap= dict({"0":"African-American", "1":"European", "2":"NA"})
OnehotVecPath='/home/procheta/vecInput.txt'


def estimateLabels(modelOutput):
    probs=[]
    labels=[]
    for i in range(modelOutput.shape[0]):
        index=np.where(modelOutput[i] == np.amax(modelOutput[i]))
        labels.append(index[0][0])
    return labels


def computeProbability(predicted_classes,outputLabel,genderLabel,gender, emotion,startIndex,indicator):
    nominator=0
    denominator=0
    count=startIndex
    count1=0
    for i in range(len(predicted_classes)):
        if (genderLabel[count]==gender and predicted_classes[i]==emotion):
            nominator=nominator+1
        if((predicted_classes[i]==emotion)):
            denominator=denominator + 1
        if (genderLabel[count]==2 and predicted_classes[i]==emotion):
            count1=count1+1
        count=count+1
    
    prob=nominator/(denominator-count1)
    if indicator == 1:
        print("Given the emotion is", emotionMap[str(emotion)], "probability of being", raceMap[str(gender)],prob)
    else:
        print("Given the emotion is", emotionMap[str(emotion)], "probability of being", genderMap[str(gender)],prob)
    return prob

def LoadGenderLabels(path):
    GenderLabel=[]
    with open(path) as fp:
        for line in fp:
            line1=line.split('\n')
            GenderLabel.append(int(line1[0]))
    return GenderLabel    

def computeposterior(input_label,epsilon):    
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
                if probs[j] <= float(epsilon):
                    biasLabel.append(0)
                else:
                    biasLabel.append(1)
                
    
    print("Probability of class 0 w.r.t input label 0",count[0]/len(input_label))
    
    print("Probability of class 1 w.r.t input label 0",count[1]/len(input_label))
    
    print("Probability of class 2 w.r.t input label 0",count[2]/len(input_label))
    
    print("Probability of class 3 w.r.t input label 0",count[3]/len(input_label))

def LoadOutputLabel(path):
    OutputLabel=[]
    with open(path) as fp:
        for line in fp:
            line1=line.split('\n')
            OutputLabel.append(int(line1[0]))
    return OutputLabel

def createTrainOutput(splitIndex, numClasses,OutputLabel):
    output=[]
    for i in range(splitIndex):
        x=np.zeros(numClasses)
        x[OutputLabel[i]]=1
        output.append(x)
    return output
def preprocessInput(passages, raceLabel, labels):
    processedInput=[]
    processedLabels=[]
    genderLabel=[]
    for i in range(len(passages)):
        if raceLabel[i] == 2 and labels[i] ==0:
            t=0
        else:
            processedInput.append(passages[i])
            processedLabels.append(labels[i])
            genderLabel.append(raceLabel[i])
    return processedInput, processedLabels,genderLabel
    
passages=[]
labels=[]
with open(OnehotVecPath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for idx in csv_reader:
        line = idx[0]
        tokens = line.split(' ')
        vec=[]
        for token in tokens:
            try:
                vec.append(float(token))
            except:
                d=0  
        passages.append(vec)
passages=np.load("/home/procheta/check_100.npy")
OutputLabel=LoadOutputLabel("/home/procheta/OutputLabel.txt")
genderLabel=LoadGenderLabels("/home/procheta/GenderLabel.txt")
genderLabel=LoadGenderLabels("/home/procheta/RaceLabel.txt")

print('Data Loaded')
print('Starting Logistic Regression Model')

length =(int) (len(passages)*.80)
#length=7000
totalLength= len(passages)
model = LogisticRegression(solver='lbfgs', penalty="l2")
model.fit(passages[0:length], OutputLabel[0:length])
predicted_classes = model.predict(passages[length:totalLength])
accuracy = accuracy_score(OutputLabel[length:totalLength],predicted_classes)

print('accuracy of the model before de-biasing', accuracy)

prob_0=computeProbability(predicted_classes,OutputLabel,genderLabel,1, 2,length,0)
prob_1=computeProbability(predicted_classes,OutputLabel,genderLabel,0, 2,length,0)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing',emotionMap[str(2)],'is',gamma)

prob_0=computeProbability(predicted_classes,OutputLabel,genderLabel,1, 0,length,0)
prob_1=computeProbability(predicted_classes,OutputLabel,genderLabel,0, 0,length,0)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing for emotion',emotionMap[str(0)],'is',gamma)

prob_0=computeProbability(predicted_classes,OutputLabel,genderLabel,1, 1,length,0)
prob_1=computeProbability(predicted_classes,OutputLabel,genderLabel,0, 1,length,0)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing',emotionMap[str(1)],'is',gamma)

prob_0=computeProbability(predicted_classes,OutputLabel,genderLabel,1, 3,length,0)
prob_1=computeProbability(predicted_classes,OutputLabel,genderLabel,0, 3,length,0)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing',emotionMap[str(3)],'is',gamma)

input_data=passages[0:length]
input_label=OutputLabel[0:length]
group_cluster=[]

for i in range(nCluster):
    x=[]
    group_cluster.append(x)


for i in range(len(input_label)):
    group_cluster[input_label[i]].append(passages[i])
    
print("First level grouping done...")

for i in range(len(group_cluster)):
    group = group_cluster[i]
    kmeans = KMeans(n_clusters=4, random_state=0).fit(group)
    computeposterior(kmeans.labels_,sys.argv[1])
    print("Clustering complete for emotion",emotionMap[str(i)])

#sys.exit()
input_label=createTrainOutput(length, 5,OutputLabel)    
def multiTaskModelWithBias(train_input,train_output,bias_output,gamma,test_input,splitIndex,numClasses,midDim):
    X = tf.placeholder("float", [splitIndex,len(train_input[0])], name="X")
    Y1 = tf.placeholder("float",[splitIndex,5], name="Y1")
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
           if i%500 == 0:
               print(Y1_loss_val," ",Y2_Loss_val)
        test_input= tf.cast(test_input, tf.float32)
        shared_layer = tf.nn.relu(tf.matmul(test_input,shared_layer_wts))
        Y1_layer_val = tf.nn.sigmoid(tf.matmul(shared_layer,Y1_layer_wts))
        Y1=session.run(Y1_layer_val)

    return Y1


biasLabel=np.asarray(biasLabel)
biasLabel=biasLabel.reshape(length,1)
Y2=multiTaskModelWithBias(input_data, input_label, biasLabel, .2,passages[length:totalLength],length,5,200)
print('bias aware model training complete')

labels=estimateLabels(Y2)
accuracy = accuracy_score(OutputLabel[length:totalLength],labels)
print('Accuracy of the model after de-biasing', accuracy)


prob_0=computeProbability(labels,OutputLabel,genderLabel,1, 2,length,0)
prob_1=computeProbability(labels,OutputLabel,genderLabel,0, 2,length,0)
fairness_score=prob_0*prob_1
print('Fairness after debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value after debiasing',gamma)


prob_0=computeProbability(labels,OutputLabel,genderLabel,1, 0,length,0)
prob_1=computeProbability(labels,OutputLabel,genderLabel,0, 0,length,0)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing for emotion',emotionMap[str(0)],'is',gamma)

prob_0=computeProbability(labels,OutputLabel,genderLabel,1, 1,length,0)
prob_1=computeProbability(labels,OutputLabel,genderLabel,0, 1,length,0)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing',emotionMap[str(1)],'is',gamma)

prob_0=computeProbability(labels,OutputLabel,genderLabel,1, 3,length,0)
prob_1=computeProbability(labels,OutputLabel,genderLabel,0, 3,length,0)
fairness_score=prob_0*prob_1
print('Fairness before debiasing approach',fairness_score)
gamma=(accuracy*fairness_score)/(accuracy+fairness_score)
print('Gamma value before debiasing',emotionMap[str(3)],'is',gamma)






