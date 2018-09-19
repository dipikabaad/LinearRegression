# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:31:13 2018

@author: Dipika
"""

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf('C:\\Dipika\\Resumes\\Siraj\\week1_0\\brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()

