import scipy.io as sio
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

dataset_1 = sio.loadmat('4_11_2019_(400pAsignal).mat')
dataset_2 = sio.loadmat('4_11_2019_(200pAnoise).mat')

neuron = 'CC'
cond = 'ctrl'

interest = neuron+'_'+cond

torch.manual_seed(0)
np.random.seed(0)
# CC control

data1 = dataset_1[neuron]['{}_f_ws'.format(cond)][0][0][200:]
data2 = dataset_2[neuron]['{}_f_ws'.format(cond)][0][0][200:]

np.random.shuffle(data1)
np.random.shuffle(data2)

train_data = np.concatenate([data1[:1800],data2[:1800]],0)
test_data = np.concatenate([data1[1800:],data2[1800:]],0)

train_labels = np.concatenate([np.ones((1800,1)), np.zeros((1800,1))], 0)
test_labels = np.concatenate([np.ones((400,1)), np.zeros((400,1))], 0)

train = np.concatenate([train_data, train_labels], 1)
test = np.concatenate([test_data, test_labels], 1)

bin_size = 100
all_data = np.concatenate([train_data,test_data],0)
min_d = int(min(all_data.flatten())*bin_size)
max_d = int(max(all_data.flatten())*bin_size)-min_d+1

def get_bin_idx(i, bin_size, min_d):
	return int(i*bin_size)-min_d

class Model(nn.Module):
	def __init__(self, embedding, hidden_dim, n_layers, batch_size):
		super(Model, self).__init__()
		self.embedding = embedding
		self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, bidirectional=True)
		self.projection = nn.Sequential(nn.Linear(in_features=hidden_dim*2, out_features=2))

	def forward(self, x):
		x = torch.tensor(x, dtype=torch.long)
		embeds = self.embedding(x)
		out, _ = self.lstm(embeds)
		output = self.projection(out)
		scores = torch.tanh(torch.sum(output,1))
		return scores

size = 180
embed_dim = 256
hidden_dim = 256
n_layers = 5
epochs = 300
batch_size = 50

embedding = nn.Embedding(max_d, embed_dim)
model = Model(embedding, hidden_dim, n_layers, batch_size)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

accuracies, signals, noises = [], [], []

for epoch in range(epochs):
	curr_time = time.time()
	epoch_loss = 0.0
	batch_idxs = random.sample(range(len(train)), batch_size)
	batch = []
	targets = []
	for i in batch_idxs:
		batch.append(list(map(lambda x: get_bin_idx(x, bin_size, min_d), train[i][:-1])))
		targets.append(train[i][-1])
	model.zero_grad()
	targets = torch.tensor(targets).long()
	scores = model(batch)
	_, pred = torch.max(scores,1)
	accuracy, signal, noise = 0, 0, 0
	for i in range(len(targets)):
		if pred[i] == targets[i]:
			accuracy += 1
			if targets[i] == 1:
				signal += 1
			else:
				noise += 1
	accuracies.append(accuracy/50)
	loss = loss_function(scores, targets)
	epoch_loss += loss
	loss.backward()
	optimizer.step()
	print('Time taken: {}s'.format(time.time()-curr_time))
	curr_time = time.time()
	print("Epoch: %d, loss: %1.5f" % (epoch, epoch_loss/len(train)))

torch.save((embedding.state_dict(), model.state_dict()), '{}_model'.format(interest))

plt.plot(accuracies, color='black')
plt.show()
plt.savefig('{}_acc.png'.format(interest), transparent=True)

# embedding_state , model_state = torch.load('{}_model'.format(interest))
# embedding.load_state_dict(embedding_state)
# model.load_state_dict(model_state)

# total = len(test)

# accuracy, signal, noise = 0, 0, 0

# batch_idxs = list(range(0,len(test),50))
# for start_idx in batch_idxs:
# 	batch = test[start_idx:start_idx+50]
# 	targets = batch[:,-1]
# 	batch = batch[:,:-1]
# 	model.zero_grad()
# 	scores = model(batch)
# 	_, pred = torch.max(scores,1)
# 	for i in range(len(targets)):
# 		if pred[i] == targets[i]:
# 			accuracy += 1
# 			if targets[i] == 1:
# 				signal += 1
# 			else:
# 				noise += 1

# file = open('classification_results.txt'.format(interest), 'a+')
# print(accuracy/8, signal/4, noise/4)
# file.write('{}\n'.format(interest))
# file.write('Total Accuracy: {}%\n'.format(accuracy/8))
# file.write('Signal Accuracy: {}%\n'.format(signal/4))
# file.write('Noise Accuracy: {}%\n'.format(noise/4))
# file.close()