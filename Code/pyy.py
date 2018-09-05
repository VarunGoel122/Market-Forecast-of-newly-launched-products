import csv,os
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('test.html',)


@app.route('/hello', methods=['POST','GET'])
def save_comment():
    if request.method == 'POST':
        Battery_Life = request.form.get('Battery_Life')
        Processor_Speeds = request.form.get('Processor_Speeds')
        Ram = request.form.get('Ram')
        Screen_Size = request.form.get('Screen_Size')
        Integrated_Wireless = request.form.get('Integrated_Wireless')
        Bundled_Applications = request.form.get('Bundled_Applications')
        Retail_Price = request.form['Retail_Price']
        HD_Size = request.form['HD_Size']
        Sales = 0
        fieldnames = ['Battery_Life','Bundled_Applications','HD_Size','Integrated_Wireless','Processor_Speeds','Ram','Retail_Price','Screen_Size','Sales']
        values = [Battery_Life, Bundled_Applications, HD_Size, Integrated_Wireless, Processor_Speeds, Ram, Retail_Price, Screen_Size,Sales]
        #values = [a,b,c,d,e,f,g,h]
        with open('Newdata.csv', 'wt', newline="") as inFile:
            writer = csv.writer(inFile, delimiter=',')
            writer.writerow(i for i in fieldnames)

            writer.writerow(j for j in values)
        #with open('test11.csv','w', newline="") as inFile:
         #   writer = csv.DictWriter(inFile, fieldnames=fieldnames)

            # writerow() will write a row in your csv file
          #  writer.writerow({'ampHrs': ampHrs,'apps': apps,'HDSize':HDSize, 'wireless':wireless, 'GHzs': GHzs, 'GBs': GBs,'retailPrice':retailPrice,'Ins':Ins})
        # And you return a text or a template, but if you don't return anything
        # this code will never work.
        return render_template('predict.html')
@app.route('/pre', methods=['POST','GET'])
def save_comment1():
    # This is to make sure the HTTP method is POST and not any other
    if request.method == 'POST':
        EcoRange = request.form['EcoRange']
        LowRange = request.form['LowRange']
        DecRange = request.form['DecRange']
        SaveRange = request.form['SaveRange']
        BenRange = request.form['BenRange']
        UseRange = request.form['UseRange']
        IdeaRange = request.form['IdeaRange']
        ClientRange = request.form['ClientRange']
        NeedRange = request.form['NeedRange']
        PotRange = request.form['PotRange']
        MechRange = request.form['MechRange']
        ResRange = request.form['ResRange']
        novel = request.form.get('novel')
        fieldnames = ['EcoRange','LowRange','DecRange','SaveRange','BenRange','UseRange','IdeaRange','ClientRange','NeedRange','PotRange','MechRange','ResRange','novel']
        with open('newtech.csv', 'w', newline="") as inFile:
            #writer = csv.DictWriter(inFile, fieldname=["Date", "temperature 1", "Temperature 2"])
            #writer.writeheader()
            writer = csv.DictWriter(inFile, fieldnames=fieldnames)
            writer.writerow({'EcoRange':EcoRange,'LowRange':LowRange,'DecRange':DecRange,'SaveRange':SaveRange,'BenRange':BenRange,'UseRange':UseRange,'IdeaRange':IdeaRange,'ClientRange':ClientRange,'NeedRange':NeedRange,'PotRange':PotRange,'MechRange':MechRange,'ResRange':ResRange,'novel':novel})
            #os.system("cd \'C:/Users/Vishwajit/myproject/flask/Scripts\';activate;cd \'C:/Users/Vishwajit/myproject/app\';python MarketForecastFinal.py")
            #os.system('python MarketForecastFinal.py')
            inFile.close()
            # Importing the libraries
            import numpy as np
            import matplotlib.pyplot as plt
            import pandas as pd

            # Importing the dataset
            dataset = pd.read_csv('Laptopsdata.csv')
            data = pd.read_csv('Newdata.csv')
            hg = np.array(data)
            dataset.loc[576, 'Battery_Life':'Sales'] = hg
            # dataset = pd.concat([dataset, data], axis=0,  ignore_index=True)
            X = dataset.iloc[:, 0:8].values
            # find the right number of clusters
            from sklearn.cluster import KMeans
            wcss = []
            for i in range(1, 20):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1, 20), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            fig1 = plt.gcf()
            plt.show()

            fig1.savefig("C:/Users/Vishwajit/myproject/app/static/graph1.png")
            # Fitting K-Means to the dataset
            kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
            y_kmeans = kmeans.fit_predict(X)
            dataset['Cluster'] = y_kmeans
            # finding cluster number of the new data
            l = dataset.iloc[575, 9]
            # Grouping by cluster Number
            g = dataset.groupby('Cluster')
            df = g.get_group(l)
            a = df.iloc[:, :-2].values
            b = df.iloc[:, 8].values
            # train test split
            a_train = df.iloc[:-1, :-2].values
            b_train = df.iloc[:-1, 8].values
            a_test = df.iloc[-1:, :-2].values
            b_test = df.iloc[-1:, 8].values
            # model fitting
            # from sklearn.cross_validation import train_test_split
            # a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.1, random_state = 0)
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
            regressor.fit(a_train, b_train)
            # Predicting the Test set results
            b_pred = regressor.predict(a_test)
            b_pred = np.round(b_pred)
            print(b_pred)
            y_new_pred = b_pred
            # from sklearn.metrics import mean_squared_error
            # from math import sqrt
            # ms = mean_squared_error(b_test, b_pred)
            # ms1 = sqrt(mean_squared_error(b_test, b_pred ))

            newtech = pd.read_csv('newtech.csv', header=None)
            newtech = np.array(newtech)
            aa1 = newtech[:, 0]
            aa2 = newtech[:, 1]
            aa3 = newtech[:, 2]
            aa4 = newtech[:, 3]
            aa5 = newtech[:, 4]
            aa6 = newtech[:, 5]
            aa7 = newtech[:, 6]
            aa8 = newtech[:, 7]
            aa9 = newtech[:, 8]
            aa10 = newtech[:, 9]
            aa11 = newtech[:, 10]
            aa12 = newtech[:, 11]
            degree = newtech[:, 12]
            relad = degree * 5 * (aa1 + aa2 + aa3 + aa4 + aa5)
            compat = degree * 4 * (aa6 + aa7 + aa8)
            complexi = degree * 6 * (aa9)
            trial = degree * 3 * aa10
            obser = degree * 4 * aa11
            maxr = 9 * 5 * 45
            maxcp = 9 * 4 * 21
            maxc = 9 * 6 * 7
            maxt = 9 * 3 * 7
            maxo = 9 * 4 * 7
            max1 = maxr + maxcp + maxc + maxt + maxo
            min1 = 5 + 4 + 4 + 3 + 4
            '''normr=(((relad-minr)/(maxr-minr))*(1.25-0.75))+0.75
            normcp=(((compat-mincp)/(maxcp-mincp))*(1.25-0.75))+0.75
            normc=(((complexi-minc)/(maxc-minc))*(1.25-0.75))+0.75
            normt=(((trial-mint)/(maxt-mint))*(1.25-0.75))+0.75
            normo=(((maxo-mino)/(maxo-mino))*(1.25-0.75))+0.75
            '''
            ip = relad + compat + complexi + trial + obser
            ip2 = (((ip - min1) / (max1 - min1)) * (1.25 - 0.75)) + 0.75

            # print(min1)
            # print(max1)
            print(ip)
            print(ip2)
            newdemand = int(y_new_pred * ip2)
            import math
            # newdemand=math.ceil(newdemand)
            print(y_new_pred)
            print(newdemand)

            import matplotlib.pyplot as plt

            diff = newdemand - y_new_pred
            nd1 = newdemand / 0.025
            x = [1, 2, 3, 4, 5]
            dds = np.array([nd1 * 0.025, nd1 * 0.135, nd1 * 0.34, nd1 * 0.34, nd1 * 0.16])
            # dds2=np.array([y_new_pred+(0.025*diff),y_new_pred+(diff*0.135),y_new_pred+(diff*0.34),y_new_pred+(diff*0.34),y_new_pred+(diff*0.16),y_new_pred])
            # dds=np.array([diff*0.025,diff*0.135,diff*0.34,diff*0.34,diff*0.16])
            # from matplotlib import pyplot as plt

            plt.plot(x, dds, marker='.', linewidth=3, markersize=20, color='blue', markerfacecolor='black',
                     linestyle='dashed', dash_joinstyle='round', dash_capstyle='round')
            plt.ylabel('SALES')
            plt.xlabel('TIME')

            plt.title('Diffusion Curve')

            # plt.plot(range(10))
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

            fig2 = plt.gcf()
            plt.show()
            fig2.savefig("C:/Users/Vishwajit/myproject/app/static/graph2.png")
            # plt.savefig('plot')
            # plt.clf()
            #print(dds2)
            # plt.xlabel()
            # from scipy.interpolate import spline

            # xnew = np.linspace(dds2.min(),dds2.max(),300) #300 represents number of points to make between T.min and T.max

            # power_smooth = spline(dds2,xnew,1)

            # plt.plot(xnew,power_smooth)
            # plt.show()
            # plt.show()

            import matplotlib.pyplot as plt;
            plt.rcdefaults()
            objects = ('Sales based on common features', 'After New features')
            y_pos = np.arange(len(objects))
            performance = [y_new_pred, newdemand]

            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('sales in number of units')
            plt.title('Demand')
            fig3 = plt.gcf()
            plt.show()
            fig3.savefig("C:/Users/Vishwajit/myproject/app/static/graph3.png")

            return render_template('result.html')
if __name__ == '__main__':
    app.run()


