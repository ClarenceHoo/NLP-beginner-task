'''
n-gram预先训练
'''
import torch 
import numpy as np
import os
import random
import torch.nn as nn
import pandas as pd
#os.environ["CUDA_VISIBLE_DEVICES"] = '4'
torch.manual_seed(1)
epoch = 10
em_dim = 100
pre_len = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
x_all = np.array(np.load('./data/n_gram_x.npy'))
y_all = np.array(np.load('./data/n_gram_y.npy'))
lenth = len(y_all)
voc_size = 80
#print(len(y_all))
#5964225 ==283*5*5*3*281


def data_process():
	df = pd.read_csv('train.tsv',header=0,delimiter='\t')
	x_train = df['Phrase']
	y_train = df['Sentiment']
	all = []
	for i in range(len(x_train)):
		all.extend(x_train[i])
	voc = set(all)
	word_to_ix = {word: i for i, word in enumerate(voc)}#构造字典
	g_x_train = []
	g_y_train = []
	for i in range(len(x_train)):
		sen_temp = [word_to_ix[i] for i in x_train[i]]
		len_temp = len(sen_temp)
		if len_temp>2:
			for j in range(len_temp-2):
				g_x_train.append((sen_temp[j],sen_temp[j+1]))
				g_y_train.append(sen_temp[j+2])
	np.save('./data/n_gram_x.npy',g_x_train)
	np.save('./data/n_gram_y.npy',g_y_train)				

	
class mlp(nn.Module):
	def __init__(self,voc_size,em_dim,pre_len):
		super(mlp,self).__init__()
		self.em = torch.nn.Embedding(voc_size,em_dim)
		self.fc1 = torch.nn.Linear(em_dim,128)
		self.fc2 = torch.nn.Linear(128,voc_size)
	def forward(self,din):
		dout = self.em(din)
		dout = torch.nn.functional.relu(self.fc1(dout))
		dout = torch.sum(dout,1)#补充求和运算
		if self.training:
			dout = torch.nn.functional.softmax(torch.nn.functional.relu(self.fc2(dout)))
		return dout
model = mlp(voc_size,em_dim,pre_len).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
batch_size = 75
itor = int(lenth/batch_size)

def train(model, device,optimizer, epoch):
	#model.train()
	for i in range(itor):
		batch_x = []
		batch_y = []
		for j in range(batch_size):
			batch_x.append(x_all[i*batch_size+j])
			batch_y.append(y_all[i*batch_size+j])
		batch_x = np.array(batch_x)
		batch_y = np.array(batch_y)
		data1 = torch.tensor(batch_x,dtype=torch.long).to(device)
		target = torch.tensor(batch_y,dtype=torch.long).to(device)
		optimizer.zero_grad()
		output = model(data1)
		loss = torch.nn.functional.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if(i+1)%1000 == 0: 
			print('Train Epoch: {} \tLoss: {:.6f}'.format(
				epoch,  loss.item()))

#测试过程
def test(model, device):
	correct =  0
	number = 100*batch_size
	with torch.no_grad():
		for i in range(100):
			batch_x = []
			batch_y = []
			for j in range(batch_size):
				batch_x.append(x_all[i*batch_size+j])
				batch_y.append(y_all[i*batch_size+j])
			batch_x = np.array(batch_x)
			batch_y = np.array(batch_y)
			data1 = torch.tensor(batch_x,dtype=torch.long).to(device)
			target = torch.tensor(batch_y,dtype=torch.long).to(device)	
			output = model(data1)
			pred = output.max(1, keepdim=True)[1] 
			correct += pred.eq(target.view_as(pred)).sum().item()
	print('Test set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
		correct, number,
		100. * correct / number))
		
def output(model, device):
	model.load_state_dict(torch.load('./model/ngram.pkl'))
	df = pd.read_csv('train.tsv',header=0,delimiter='\t')
	x_train = df['Phrase']
	y_train = df['Sentiment']
	all = []
	for i in range(len(x_train)):
		all.extend(x_train[i])
	voc = set(all)
	word_to_ix = {word: i for i, word in enumerate(voc)}#构造字典
	features = []
	model.eval()
	with torch.no_grad():
		for i in range(len(x_train)):
			sen_temp = [word_to_ix[i] for i in x_train[i]]			
			data = torch.tensor(sen_temp,dtype=torch.long).to(device)
			data = torch.unsqueeze(data, 0) 
			output = torch.squeeze(model(data)).data.cpu().numpy()
			#print(output.shape)

			features.append(output)	
	np.save('./data/ngram_featrue_x.npy',features)

def model_get():
	for epoch in range(1, epoch + 1):
		train(model, DEVICE, optimizer, epoch)
		test(model, DEVICE)
	torch.save(model.state_dict(), './model/ngram.pkl')

total = sum([param.nelement() for param in model.parameters()])
print('  + Number of params: %.2fM' % (total / 1e6))
model_get()
output(model, DEVICE)
