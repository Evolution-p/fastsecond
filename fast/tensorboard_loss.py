#import tensorflow as tf
#from numpy import random

#writer_1 = tf.summary.FileWriter("./logs/plot_1")
#writer_2 = tf.summary.FileWriter("./logs/plot_2")

#log_var = tf.Variable(0.0)
#tf.summary.scalar("loss", log_var)

#write_op = tf.summary.merge_all()

#session = tf.InteractiveSession()
#session.run(tf.global_variables_initializer())

#for i in range(200):
    # for writer 1
    #summary = session.run(write_op, {log_var: random.rand()})
    #writer_1.add_summary(summary, i)
    #writer_1.flush()

    # for writer 2
    #summary = session.run(write_op, {log_var: random.rand()})
    #writer_2.add_summary(summary, i)
    #writer_2.flush()


import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

net2 = pd.read_csv('./runs/loss/att_unet_plus_Train_Loss.csv', usecols=['Step', 'Value'])
plt.plot(net2.Step, net2.Value, lw=1.5, label='Att_Nested_Unet', color='pink')
net3 = pd.read_csv('./runs/loss/att_unet_Train_Loss.csv', usecols=['Step', 'Value'])
plt.plot(net3 .Step, net3 .Value, lw=1.5, label='Att_Unet', color='green')
net4 = pd.read_csv('./runs/loss/unet_plus_Train_Loss.csv', usecols=['Step', 'Value'])
plt.plot(net4 .Step, net4 .Value, lw=1.5, label='Unet++', color='yellow')
net5 = pd.read_csv('./runs/loss/unet_Train_Loss.csv', usecols=['Step', 'Value'])
plt.plot(net5 .Step, net5 .Value, lw=1.5, label='Unet', color='red')
plt.legend(loc=0)
plt.show()
