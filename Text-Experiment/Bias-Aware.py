import numpy as np
import math
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
import sys


#Parameter Declaration

inputVecpath="/home/procheta/vecInput.txt"
labelPath="/home/procheta/OutputLabel.txt"
raceLabelPath="/home/procheta/RaceLabel.txt"
genderLabelPath="/home/procheta/GenderLabel.txt"
genderMap = dict({"1":"female", "0":"male"})
emotionMap= dict({"0":"anger", "1":"sadness", "2":"fear", "3":"joy", "4":"no_emotion"})
raceMap= dict({"0":"African-American", "1":"European", "2":"NA"})
nClass=5
nEpoch=2000
dimension=200
emotionLabel=0
indicator=1

def loadInputVec(path):
    input_vec=[]
    with open(path) as fp:
        for line in fp:
            line1=line.split('\n')
            words = line1[0].split(' ')
            z=[]
            for word in words:
                z.append(float(word))
            input_vec.append(z)
    return input_vec

def LoadAttributeLabels(path):
    OutputLabel=[]
    with open(path) as fp:
        for line in fp:
            line1=line.split('\n')
            OutputLabel.append(int(line1[0]))
    return OutputLabel

def computeProbability(predicted_classes,genderLabel,gender,emotion):
    nominator=0
    denominator=0
    for i in range(len(predicted_classes)):
        if (genderLabel[i]==gender and predicted_classes[i]==emotion):
            nominator=nominator+1
        if((predicted_classes[i]==emotion)):
            denominator=denominator + 1
    prob=nominator/denominator
    return prob
        

def createTrainOutput(splitIndex, numClasses,OutputLabel):
    output=[]
    for i in range(splitIndex):
        x=np.zeros(numClasses)
        x[OutputLabel[i]]=1
        output.append(x)
    return output

#estimating labels from multi-task learning classifier output
def estimateLabels(modelOutput):
    probs=[]
    labels=[]
    for i in range(modelOutput.shape[0]):
        index=np.where(modelOutput[i] == np.amax(modelOutput[i]))
        labels.append(index[0][0])
    return labels

def createBiasOutput(OutputLabel,genderLabel,splitIndex, output_label):
    bias_output=[]
    for i in range(splitIndex):
        if OutputLabel[i] == output_label and genderLabel[i]==1 :
            bias_output.append(1)
        else:
            bias_output.append(0)
    return bias_output

def multiTaskModelWithBias(train_input,train_output,constant_matrix,bias_output,test_input,midDim):
    X = tf.placeholder("float", [len(train_input),len(train_input[0])], name="X")
    Y1 = tf.placeholder("float",[len(train_input),nClass], name="Y1")
    Y2 = tf.placeholder("float", [len(train_input),1], name="Y2")

    tf.set_random_seed(1236)
    np.random.seed(1234)

    initial_shared_layer_weights = np.random.rand(len(train_input[0]),midDim)
    initial_Y1_layer_weights = np.random.rand(midDim, nClass)
    initial_Y2_layer_weights = np.random.rand(midDim,1)

    shared_layer_weights = tf.Variable(initial_shared_layer_weights, name="share_W", dtype="float32")
    Y1_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y1", dtype="float32")
    Y2_layer_weights = tf.Variable(initial_Y2_layer_weights, name="share_Y2", dtype="float32")

    shared_layer = tf.nn.relu(tf.matmul(X,shared_layer_weights))
    Y1_layer = tf.nn.sigmoid(tf.matmul(shared_layer,Y1_layer_weights))


    subspace_layer=tf.multiply(shared_layer,constant_matrix)
    Y2_layer = tf.nn.sigmoid(tf.matmul(subspace_layer,Y2_layer_weights))

    # Calculate Loss
    Y1_Loss = tf.nn.l2_loss(Y1-Y1_layer)
    Y2_Loss = tf.nn.l2_loss(Y2-Y2_layer)
    Y2_Loss = -Y2_Loss

    
    Y1_op = tf.train.AdamOptimizer().minimize(Y1_Loss)
    Y2_op = tf.train.AdamOptimizer().minimize(Y2_Loss)
    Y1_layer_wts=0
    shared_layer_wts=0
    Y1_layer_val =0
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        for i in range(nEpoch):
           _,tt,Y1_loss_val, Y2_Loss_val, shared_layer_wts, Y1_layer_wts,Y2_layer_wts = session.run([Y1_op,Y2_op,Y1_Loss, Y2_Loss, shared_layer_weights, Y1_layer_weights, Y2_layer_weights],{X: train_input,Y1: train_output,Y2: bias_output})
           if i%100 == 0:
               print(Y1_loss_val," ",Y2_Loss_val)
        test_input= tf.cast(test_input, tf.float32)
        shared_layer = tf.nn.relu(tf.matmul(test_input,shared_layer_wts))
        Y1_layer_val = tf.nn.sigmoid(tf.matmul(shared_layer,Y1_layer_wts))
        Y1=session.run(Y1_layer_val)

    return Y1


def computeConstantMatrix(length,primaryTaskLabel,dimension):
    constant_matrix=[]
    for i in range(length):
        x=[]
        if OutputLabel[i] == primaryTaskLabel:
            for j in range(dimension):
                x.append(np.float32(1))
        else:
            for j in range(dimension):
                x.append(np.float32(0))
        constant_matrix.append(x)
    
    return constant_matrix




input_vec=loadInputVec(inputVecpath)
OutputLabel=LoadAttributeLabels(labelPath)
if indicator == 0:
    attributeLabel=LoadAttributeLabels(raceLabelPath)
else:
    attributeLabel=LoadAttributeLabels(genderLabelPath)

length=len(input_vec)*.80
totalLength= len(input_vec)
length=7000

model = LogisticRegression(solver='lbfgs', penalty="l2")
model.fit(input_vec[0:length], OutputLabel[0:length])
predicted_classes = model.predict(input_vec[length:totalLength])
accuracy = accuracy_score( OutputLabel[length:totalLength],predicted_classes)
print("Accuracy of the Logistic Regression Classifier Before Debiasing: ",accuracy)


prob_0=computeProbability(predicted_classes,attributeLabel[length:totalLength],1, emotionLabel)
prob_1=computeProbability(predicted_classes,attributeLabel[length:totalLength],0, emotionLabel)
fairness=prob_0*prob_1
print("Fairness value for emotion", emotionMap[str(emotionLabel)], "is: ", fairness)
gamma=(fairness*accuracy)/(fairness + accuracy)
print("Gamma value for emotion", emotionMap[str(emotionLabel)], "is: ", gamma)



train_output=createTrainOutput(length, nClass,OutputLabel)
bias_output=createBiasOutput(OutputLabel,attributeLabel,length,emotionLabel)
bias_output=np.asarray(bias_output)
bias_output=bias_output.reshape(length,1)
constant_matrix=computeConstantMatrix(length,emotionLabel,dimension)

Y2=multiTaskModelWithBias(input_vec[0:length], train_output, constant_matrix, bias_output,input_vec[length:totalLength],dimension)
labels=estimateLabels(Y2)
accuracy = accuracy_score(OutputLabel[length:totalLength],labels)
print( "Multi task learning Classifier Accuracy ", accuracy)
prob_0=computeProbability(predicted_classes,attributeLabel[length:totalLength],1, emotionLabel)
prob_1=computeProbability(predicted_classes,attributeLabel[length:totalLength],0, emotionLabel)

fairness=prob_0*prob_1
print("Fairness value for emotion", emotionMap[str(emotionLabel)], "is: ", fairness)
gamma=(fairness*accuracy)/(fairness + accuracy)
print("Gamma value for emotion", emotionMap[str(emotionLabel)], "is: ", gamma)

