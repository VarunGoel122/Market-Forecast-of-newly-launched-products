import pandas as pd
import numpy as np
laptops=pd.read_csv('Laptop (2).csv')
y=laptops.iloc[:,8]
x=laptops.iloc[:,:-1]
y.head()
x.head()

from sklearn.cross_validation import train_test_split
x_train, x_test,y_train, y_test= train_test_split(x,y,test_size=0.1,random_state=1)

from sklearn.ensemble import RandomForestRegressor
Regressor= RandomForestRegressor(n_estimators=400,random_state=1)
Regressor.fit(x_train,y_train)

ynewtest=pd.read_csv('test11.csv',header=None)
ynewtest=np.array(ynewtest)
#y_new_pred=int(Regressor.predict([[5,0,120,1,2,4,420,14]]))
y_new_pred=int(Regressor.predict(ynewtest))
newtech=pd.read_csv('newtech.csv',header=None)
newtech=np.array(newtech)
aa1=newtech[:,0]
aa2=newtech[:,1]
aa3=newtech[:,2]
aa4=newtech[:,3]
aa5=newtech[:,4]
aa6=newtech[:,5]
aa7=newtech[:,6]
aa8=newtech[:,7]
aa9=newtech[:,8]
aa10=newtech[:,9]
aa11=newtech[:,10]
aa12=newtech[:,11]
degree=newtech[:,12]

relad=degree*5*(aa1+aa2+aa3+aa4+aa5)


compat=degree*4*(aa6+aa7+aa8)

complexi=degree*6*(aa9)
trial=degree*3*aa10
obser=degree*4*aa11
maxr=9*5*45
maxcp=9*4*21
maxc=9*6*7
maxt=9*3*7
maxo=9*4*7

max1=maxr+maxcp+maxc+maxt+maxo
minr=5
mincp=4
minc=4
mint=3
mino=4

min1=minr+mincp+minc+mint+mino
normr=(((relad-minr)/(maxr-minr))*(1.25-0.75))+0.75
normcp=(((compat-mincp)/(maxcp-mincp))*(1.25-0.75))+0.75
normc=(((complexi-minc)/(maxc-minc))*(1.25-0.75))+0.75
normt=(((trial-mint)/(maxt-mint))*(1.25-0.75))+0.75
normo=(((maxo-mino)/(maxo-mino))*(1.25-0.75))+0.75

ip=relad+compat+complexi+trial+obser
ip2=(((ip-min1)/(max1-min1))*(1.25-0.75))+0.75

#print(min1)
#print(max1)
print(ip)
print(ip2)
newdemand=int(y_new_pred*ip2)
import math
newdemand=math.ceil(newdemand)
print(y_new_pred)
print(newdemand)

import matplotlib.pyplot as plt
newdem2=newdemand+(0.025*newdemand)
newdem3=newdemand+(0.135*newdemand)
newdem4=newdemand+(0.34*newdemand)
newdem5=newdemand+(0.34*newdemand)
newdem6=newdemand+(0.16*newdemand)
x=[1,2,3,4,5,6,7,8]
dds=np.array([y_new_pred,newdemand,newdem2,newdem3,newdem4,newdem5,newdem6,y_new_pred])


plt.plot(x,dds,marker='.',linewidth=3,markersize=20,color='blue',markerfacecolor='black',linestyle='dashed')

plt.show()
plt.savefig('finalgraph.png')

