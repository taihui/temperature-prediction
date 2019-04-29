import matplotlib.pyplot as plt



simpleRNN = [3.5,4.5,3.1]
LSTM = [0.5,0.8,0.9]
plt.plot(range(1,4),simpleRNN)
plt.plot(range(1,4),LSTM)
plt.ylabel("Temperture Error")
legend = ['SimpleRNN', 'LSTM']
#plt.xticks(range(0, 2))
plt.yticks(range(1, 7))
plt.legend(legend)


plt.title('Prediction Performance')
plt.show()