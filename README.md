# Market Forecast of newly launched products

This is a Flask based web app to allow companies to forecast the demand of the newly launched laptops, taking in consideration various market factors & conditions and the innovation factor with some basic specifications of laptops.

Data-
The data we have is of existing laptops with their specifications and sales. Around 580 laptops with different specifications. 

Libraries & Machine Learning Algorithms-
1. Numpy
2. Pandas
3. Matplotlib
4. Scikit-Learn
5. Flask
6. K-Means Clustering
7. Linear Regression
8. Random Forest Regressor

Methodology-
1. Historical data is present with laptop specifications and their sales.
2. The client enters the specifications of the new laptop. This data is clustered with historical data using K-Means to find the existing laptops with similar specifications. 
3. The Multi-Linear Regression model is trained using the sales of similar existing laptops to find the sales of new product without innovation factor.
4. Now, the innovation factor is calculated on the basis of various market conditions. A normalized innovation potential is calculated between 0.75 - 1.25. If the potential is below 1, the sales will decrease and if it is above 1, sales will increase.
5. Degree of novelty represents that if the innovation is existing in the company, new in the company or new in the market.
6. Basic architecture of the project and the innovation potential is provided in the documentation folder. 

Future Work-
1. We can take into consideration many other market factors such as festive season, holidays etc. which can also affect the sales of the product.
2. We plan on implementing Deep Learning algorithms such as Restricted Boltzmann Machine or RNN for better prediction of sales.
3. We can also try finding the sales of the new product region wise.
