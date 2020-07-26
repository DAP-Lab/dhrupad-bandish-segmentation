import os
import sys
import glob
import torch
from torch.utils import data
from model_utils import *
from params import *
import numpy as np
import matplotlib.pyplot as plt

##generators
training_set = Dataset(datadir, partition['train'], labels_stm)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(datadir, partition['validation'], labels_stm)
validation_generator = data.DataLoader(validation_set, **params)

#model definition and training
model=build_model(input_height,input_len,n_classes).float().to(device)
criterion=torch.nn.CrossEntropyLoss(reduction='mean')
optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)

print(model)
n_params=0
for param in model.parameters():
		n_params+=torch.prod(torch.tensor(param.shape))
print('No of trainable params: %d\n'%n_params)

##training epochs loop
train_loss_epoch=[]; train_acc_epoch=[]
val_loss_epoch=[]; val_acc_epoch=[]
n_idle=0
for epoch in range(max_epochs):
	if n_idle==50: break
	train_loss_epoch+=[0]; train_acc_epoch+=[0]
	val_loss_epoch+=[0]; val_acc_epoch+=[0]
	y_pred_all=np.array([]); y_true_all=np.array([])

	n_iter=0
	##training
	model.train()
	for local_batch, local_labels, _ in training_generator:
		#map labels to class ids
		local_labels=class_to_categorical(local_labels,classes)

		#add channel dimension
		if len(local_batch.shape)==3: local_batch = local_batch.unsqueeze(1)
		
		#transfer to GPU
		local_batch, local_labels = local_batch.float().to(device), local_labels.to(device)

		#update weights
		optimizer.zero_grad()
		outs = model(local_batch).squeeze()
		loss = criterion(outs, local_labels.long())
		loss.backward()
		optimizer.step()
		
		#append loss and acc to arrays
		train_loss_epoch[-1]+=loss.item()
		acc = np.sum((np.argmax(outs.detach().cpu().numpy(),1)==local_labels.detach().cpu().numpy()))/batch_size
		train_acc_epoch[-1]+=acc
		n_iter+=1

	train_loss_epoch[-1]/=n_iter
	train_acc_epoch[-1]/=n_iter

	n_iter=0
	##validation
	model.eval()
	with torch.set_grad_enabled(False):
		for local_batch, local_labels, example_ids in validation_generator:
			#map labels to class ids
			local_labels=class_to_categorical(local_labels,classes)

			#add channel dimension
			if len(local_batch.shape)==3: local_batch = local_batch.unsqueeze(1)

			#transfer to GPU
			local_batch, local_labels = local_batch.float().to(device), local_labels.to(device)
			
			#evaluate model
			outs = model(local_batch).squeeze()
			loss = criterion(outs, local_labels.long())

			#append loss and acc to arrays
			val_loss_epoch[-1]+=loss.item()
			acc = np.sum((np.argmax(outs.detach().cpu().numpy(),1)==local_labels.detach().cpu().numpy()))/batch_size
			val_acc_epoch[-1]+=acc

			n_iter+=1
		val_loss_epoch[-1]/=n_iter
		val_acc_epoch[-1]/=n_iter

	#save if val_loss reduced
	if val_loss_epoch[-1]==min(val_loss_epoch): 
		torch.save(model.state_dict(), os.path.join(model_dir,mode, 'saved_model_fold_%d.pt'%fold))
		n_idle=0
	else: n_idle+=1
	
	#print loss in current epoch
	print('Epoch no: %d/%d\tTrain loss: %f\tTrain acc: %f\tVal loss: %f\tVal acc: %f'%(epoch, max_epochs, train_loss_epoch[-1], train_acc_epoch[-1],val_loss_epoch[-1],val_acc_epoch[-1]))
	
	#plot losses vs epoch
	plt.plot(train_loss_epoch,label='train')
	plt.plot(val_loss_epoch,label='val')
	plt.legend()
	plt.savefig(os.path.join(plot_dir,mode,'loss_curves_fold_%d.png'%fold))
	plt.clf()
