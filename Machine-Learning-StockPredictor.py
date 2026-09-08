import numpy as np
import random
import yfinance as yf
import matplotlib.pyplot as plt
import requests


url='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&apikey=QS8S66JLOPKE1LII'
r=requests.get(url)
data=r.json()


numoftrees=10
minsamples=2
maxdepth=5

# WHEN API REQUESTS FAIL OR LIMIT IS REACHED code below will no longer work

sp500 = yf.Ticker("^GSPC").history(period="1mo")
opening_prices = sp500['Open'].values
closing_prices = sp500['Close'].values
sp500 = sp500.drop(columns=["Volume","High","Low"])
sp500['Target']=sp500['Close'].shift(-1)
sp500=sp500[:-1]
result=sp500[['Open','Close','Target']].to_numpy()
datasetog=np.round(result,decimals=2)



time_series = data["Time Series (Daily)"]
sorted_dates = sorted(time_series.keys())
datasetog = []

for i in range(len(sorted_dates) - 1):
    current_day = time_series[sorted_dates[i]]
    next_day = time_series[sorted_dates[i + 1]] 
    
    open_price = float(current_day["1. open"])
    close_price = float(current_day["4. close"])
    next_close_price = float(next_day["4. close"])
    
    datasetog.append([open_price, close_price, next_close_price])

dataTestog=datasetog[-20:]
targetreal=np.array(dataTestog)[:,2]

datatest=np.array(dataTestog)[:,:2]
datasetog=datasetog[:80]


#Decision Trees

class Node():
    def __init__(self,feature=None,threshold=None,left=None,right=None,MSEvalue=None,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.MSEvalue=MSEvalue

        self.value=value

class Tree():
    def __init__(self,minsamples,maxdepth):
        self.root=None
        self.minsamples=minsamples
        self.maxdepth=maxdepth
    
    def buildtree(self,datasetog,currentdepth=0):
        dataset=np.array(datasetog)
        x,y=dataset[:,:2],dataset[:,-1]
        numsamples,numfeatures=np.shape(x)
        if numsamples>=self.minsamples and currentdepth<=self.maxdepth:
            bestsplit=self.bestsplit(dataset,numsamples,numfeatures)
            if bestsplit["MSEvalue"]>0:
                leftsubtree=self.buildtree(bestsplit["datasetleft"],currentdepth+1)
                rightsubtree=self.buildtree(bestsplit["datasetright"],currentdepth+1)
                
                return Node(bestsplit['feature'],bestsplit['point'],leftsubtree,rightsubtree,bestsplit['MSEvalue'])
        leafvalue=self.calculateleafvalue(y)
        return Node(value=leafvalue)
    
    def calculateleafvalue(self,y):
        value=np.mean(y)
        return value

    def split(self,dataset,feature,point):
        #splitting on feature and criteria
        datasetleft=np.array([row for row in dataset if row[feature]<=point])
        datasetright=np.array([row for row in dataset if row[feature]>point])
        return datasetleft,datasetright

    def bestsplit(self,dataset,numsamples,numfeatures):
        bestsplit={}
        bestmse=(float('inf'))
        for feature in range(numfeatures):
            featurevalues=dataset[:,feature]
            possiblesplitpoints=np.unique(featurevalues)
            for point in possiblesplitpoints:
                datasetleft,datasetright=self.split(dataset,feature,point)
                if len(datasetleft)>0 and len(datasetright)>0:
                    y,lefty,righty=dataset[:,-1],datasetleft[:,-1],datasetright[:,-1]
                    currentMSE=self.MSEcalc(y,lefty,righty)
                    if currentMSE<bestmse or bestmse<0:
                        bestsplit["feature"]=feature
                        bestsplit["point"]=point
                        bestsplit["datasetleft"]=datasetleft
                        bestsplit["datasetright"]=datasetright
                        bestsplit["MSEvalue"]=currentMSE
                        bestmse=currentMSE
                else:
                    continue
        if len(bestsplit)==0:
            bestsplit["feature"]=feature
            bestsplit["point"]=point
            bestsplit["datasetleft"]=datasetleft
            bestsplit["datasetright"]=datasetright
            bestsplit["MSEvalue"]=(float('-inf'))
        return bestsplit
    
    def MSEcalc(self,y,lefty,righty):
        overallmean=np.mean(y)
        
        leftmean = np.mean(lefty) if len(lefty) > 0 else 0
        rightmean = np.mean(righty) if len(righty) > 0 else 0

        mse_left = np.mean((lefty - leftmean) ** 2) if len(lefty) > 0 else 0
        mse_right = np.mean((righty - rightmean) ** 2) if len(righty) > 0 else 0
        
        weightedfinalmse=(len(lefty) * mse_left + len(righty) * mse_right) / len(y)
        
        return weightedfinalmse
    
    def starttree(self,datasetog):
        self.root=self.buildtree(datasetog)
    
    def makeprediction(self,x,tree):
        if tree.value!=None:
            return tree.value
        featureval=x[tree.feature]
        if featureval<=tree.threshold:
            return self.makeprediction(x,tree.left)
        else:
            return self.makeprediction(x,tree.right)
    
    def startprediction(self,datatest):
        predictions=[self.makeprediction(datatest,self.root)]
        return predictions


#Random forest implementation

def main():
    predictionsaverage=[]
    for j in range(0,len(datatest)):
        predictions=[]
        for tree in range(0,numoftrees):
            tree=None
            currenttree=[]
            for day in range(1,len(datasetog)):
                dayvalue=random.randint(0,len(datasetog)-1)
                currenttree.append(datasetog[dayvalue])
            tree=Tree(minsamples,maxdepth)
            tree.starttree(currenttree)
            predictions.append(tree.startprediction(datatest[j]))
        predictionsaverage.append(np.mean(predictions))

    mape=np.mean(np.abs((targetreal-predictionsaverage)/targetreal)*100)
    print(f"The prediction percentage error on average is {mape}%")

    #Plot predictions vs actual data matplotlib
    plt.plot(targetreal)
    plt.plot(predictionsaverage)
    plt.legend(["Real Values","Machine Learning Prediction"])
    plt.xticks(ticks=range(len(targetreal)),labels=(f'Day {i+1}' for i in range(0,len(targetreal))),rotation=90)
    plt.title("Predicting Past 20 days on SNP500")
    plt.show()

main()