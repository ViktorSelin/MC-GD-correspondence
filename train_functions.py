# Imports
import numpy as np
import torch
import torch.nn as nn
import time
import copy


"""# Gradient descent training function"""

#####
# Main training function for clipped gradient descent
#####
# model: neural network to be trained
# data: dictionary contain data for training in the form: data['x'], data['y'], has shape (n,1) for n data points
# n_epochs: number of epochs of training
# learning_rate: learning rate
# limit: loss at which to stop training; None means train for all epochs
# skip: number of epochs between data output
#####
# returns a dictionnary of all tracked values
# contains 'epoch', 'loss', 'params', 'time'
#####
def train_clipped_GD(model, data, n_epochs=100, learning_rate=0.001,limit=None,skip=100):
  #-----------------
  #Setup Training Data
  #-----------------
  x_tensor = torch.from_numpy(data['x']).float()
  y_tensor = torch.from_numpy(data['y']).float()
  
  #MSE loss function
  loss_func = nn.MSELoss()

  #Set up lists for plotting
  losses = []
  parameters = []
  epochs = []

  t0 = time.time()
  #-----------------
  #Calculate initial loss
  #-----------------     
  prediction = model(x_tensor)
  old_loss = loss_func(prediction,y_tensor).item()
  print('Initial_loss: %g'%(old_loss))



  #-----------------
  #Save initial parameters
  #----------------- 
  current_parameters = np.array([])
  for name,p in model.named_parameters():
    current_parameters = np.concatenate((current_parameters,p.detach().clone().numpy().reshape(-1)))
  parameters.append(current_parameters)
  #Save initial loss
  losses.append(old_loss)
  epochs.append(0)
  #-----------------
  #Train for n_epochs or until loss < limit
  #-----------------
  for epoch in range(1,n_epochs+1):
    #Calculate loss
    prediction = model(x_tensor)
    loss = loss_func(prediction,y_tensor)
    loss_item = loss.item()
    
    #Gradient descent
    loss.backward()

    with torch.no_grad():
      #Calculate |grad U|
      W_grad_squared_sum = 0
      for name, W in model.named_parameters():
        W_grad_squared_sum += torch.sum(torch.square(W.grad))
      W_grad_squared_sum = torch.sqrt(W_grad_squared_sum)

      #Perform clipped gradient descent including |grad U|
      for name, W in model.named_parameters():
        W -= 1/(W_grad_squared_sum) * learning_rate * W.grad
        W.grad.zero_()

    if epoch%skip == 0:
      current_parameters = np.array([])
      for name,p in model.named_parameters():
        current_parameters = np.concatenate((current_parameters,p.detach().clone().numpy().reshape(-1)))
      parameters.append(current_parameters)
      losses.append(loss_item)
      epochs.append(epoch)
    #-----------------
    #Stop training if limit is reached
    #-----------------
    if limit is not None and loss_item < limit:
      break

    #Print progress
    if epoch%(n_epochs//10)==0:
      #clear_output(wait=True)
      print("Progress=%g%%, Loss=%g"%(epoch/n_epochs*100,loss_item))

  t1 = time.time()
  training_time = t1-t0
  print('Final loss:%g, Training time:%gs'%(loss_item,training_time))
  vals = {'loss':np.array(losses),'params':np.array(parameters),'time':np.array([training_time]),'epoch':np.array(epochs)} 
  return vals

"""# MC training function"""
#####
# Main training function for gradient descent using a zero-temperature Metropolis Monte Carlo (MC) algorithm, a simple form of neuroevolution
# Trains a given network on given data using either vanilla MC
#####
# model: neural network to be trained
# data: dictionary contain data for training in the form: data['x'], data['y'], has shape (n,1) for n data points
# n_epochs: number of epochs of training
# elr: effective learning rate, the parameter "lambda" described in the paper
# limit: loss at which to stop training; None means train for all epochs
# skip: number of epochs between data output
#####
# returns a dictionnary of all tracked values
# contains 'epoch', 'loss', 'params', 'time'
#####
def train_MC(model, data, n_epochs=100,elr=0.1,limit=None,skip=100):
  #Calculate basic step size sigma as a function of learning rate, per Eq. (54) of the paper
  sigma = elr*np.sqrt(2*np.pi)
  print("Sigma:%g"%(sigma))
  #-----------------
  #Setup Training Data
  #-----------------
  x_tensor = torch.from_numpy(data['x']).float()
  y_tensor = torch.from_numpy(data['y']).float()

  #MSE loss function
  loss_func = nn.MSELoss()

  #Set up lists for plotting
  losses = []
  parameters = []
  epochs = []

  t0 = time.time()
  torch.set_grad_enabled(False)

  #-----------------
  #Calculate initial loss
  #-----------------      
  prediction = model(x_tensor)
  loss = loss_func(prediction,y_tensor)
  old_loss = loss.item()
  print('Initial_loss: %g'%(old_loss))

  #-----------------
  #Save initial parameters
  #----------------- 
  #Save zeroth epoch
  epochs.append(0)
  #Save Loss
  losses.append(old_loss)
  #Save Parameters
  current_parameters = np.array([])
  for name,p in model.named_parameters():
    current_parameters = np.concatenate((current_parameters,p.detach().clone().numpy().reshape(-1)))
  parameters.append(current_parameters)


  #-----------------
  #Train for n_epochs or until loss < limit
  #-----------------
  for epoch in range(1,n_epochs+1):
    #Old parameters
    old_par = copy.deepcopy(model.state_dict())


    #Update model parameters with Gaussian
    for name,p in model.named_parameters():
      p += torch.empty(p.size()).normal_(mean=0,std=sigma)

    #Calculate loss with updated parameters
    prediction = model(x_tensor)
    loss = loss_func(prediction,y_tensor)
    new_loss = loss.item()

    #-----------------
    #Accept or reject updated weights
    #-----------------
    #Accept if new loss is smaller
    if new_loss <= old_loss:
      old_loss = new_loss

    else:
      #Keep old parameters
      model.load_state_dict(old_par)
      new_loss = old_loss

    #Save every skip epochs
    if epoch%skip == 0:
      current_parameters = np.array([])
      for name,p in model.named_parameters():
        current_parameters = np.concatenate((current_parameters,p.detach().clone().numpy().reshape(-1)))
      parameters.append(current_parameters)
      losses.append(new_loss)
      epochs.append(epoch)

    #-----------------
    #Stop training if limit is reached
    #-----------------
    if limit is not None and new_loss < limit:
      break

    #Print progress
    if epoch%(n_epochs//10)==0:
      print("Progress=%g%%, Loss=%g"%(epoch/n_epochs*100,new_loss))

  t1 = time.time()
  training_time = t1-t0
  print('Final loss:%g, Training time:%gs'%(new_loss,training_time))
  vals = {'loss':np.array(losses),'params':np.array(parameters),'time':np.array([training_time]),'epoch':np.array(epochs)} 
  return vals



#####
'''
Example of how to run the training functions
How to run:
python train_functions.py
'''
#####
if __name__ == "__main__":
  
  #Import model
  from model import Net


  #Generate initial model and save initial parameters
  model = Net(1,40,1,1)
  initial_param = model.state_dict()

  #Load data
  data = np.load('./Data/sin(1pix).npz')

  '''
  #Example on how to generate data:
  #x = np.linspace(-1,1,1001).reshape(-1,1)
  #y = np.sin(np.pi*x)
  #np.savez('./Data/example_data',x=x,y=y)
  '''

  #Select some hyperparameters
  n_epochs = 10000
  gd_lr = 0.0001

  #Test GD
  print("clipped GD")
  model = Net(1,40,1,1)
  model.load_state_dict(initial_param)
  vals = train_clipped_GD(model,data,n_epochs=n_epochs, learning_rate=gd_lr,limit=None,skip=n_epochs//200)
  #save vals
  np.savez('Trajectories/gd_traj',**vals)

  #Run for a number of MC trajectories (default=10)
  n_traj = 10

  #Test MC
  for n in range(n_traj):
    print("\nMC Traj %g/%g"%(n+1,n_traj))
    model = Net(1,40,1,1)
    model.load_state_dict(initial_param)
    vals = train_MC(model,data,n_epochs=n_epochs,elr=gd_lr,limit=None,skip=n_epochs//200)
    #save vals
    np.savez('Trajectories/mc_traj_%g'%(n),**vals)

  
  #Plot, set to False to not plot
  Plot_traj = True

  if Plot_traj:
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams['axes.linewidth'] = 4

    #Calculate mean MC trajectory
    for n in range(n_traj):
      traj = np.load('Trajectories/mc_traj_%g.npz'%(n))
      mc_time = traj['epoch']*gd_lr
      if n == 0:
        mean_loss = traj['loss']
        mean_param = traj['params']
      else:
        mean_loss += traj['loss']
        mean_param += traj['params']
    mean_loss /= n_traj
    mean_param /= n_traj

    #Load GD
    gd_file = np.load('Trajectories/gd_traj.npz')
    gd_time = gd_file['epoch']*gd_lr
    ######
    #Plot loss
    ######
    print("\nPlotting Loss")
    fig,ax = plt.subplots(figsize=(10,10))
    #Plot GD
    ax.plot(gd_time,gd_file['loss'],color=(0,0,0),zorder=0,label='gradient',linewidth = 4,ls=':')
    #Plot MC traj
    for n in range(n_traj):
      traj = np.load('Trajectories/mc_traj_%g.npz'%(n))
      if n == 0:
        ax.plot(mc_time,traj['loss'],color=(0.6,0.6,0.6),alpha=0.8,zorder=1,label='evolution',linewidth = 3)
      else:
        ax.plot(mc_time,traj['loss'],color=(0.6,0.6,0.6),alpha=0.5,zorder=1,linewidth = 3)

    #Plot mean MC traj
    ax.plot(mc_time,mean_loss,color=(0,0.8,0),zorder=2,label='<evolution>',linewidth = 4)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_xlabel('$\it{t}$',fontsize=34)
    ax.set_ylabel('$\it{U}$',fontsize=34,labelpad=20)
    ax.tick_params(which='major',direction='in',width=4,length=10,labelsize=30,pad = 10)
    ax.tick_params(which='minor',direction='in',width=4,length=0,labelsize=30)
    ax.legend(fontsize=24,fancybox=False,edgecolor=(0,0,0),framealpha=1,loc='upper right')
    fig.tight_layout()
    fig.savefig("Plots/Loss_evo.png",dpi=300)

    #####
    #Plot some params, the files contain all params, 4 are chosen at even spacing over all parameters
    #####
    n_params = np.shape(mean_param)[1]
    for param_i in range(0,n_params,n_params//4):
      print("Plotting Param %g"%(param_i))
      fig,ax = plt.subplots(figsize=(10,10))
      ax.plot(gd_time,gd_file['params'][:,param_i],color=(0,0,0),zorder=0,label='gradient',linewidth = 4,ls=':')
      #Plot MC traj
      for n in range(n_traj):
        traj = np.load('Trajectories/mc_traj_%g.npz'%(n))
        if n == 0:
          ax.plot(mc_time,traj['params'][:,param_i],color=(0.6,0.6,0.6),alpha=0.8,zorder=1,label='evolution',linewidth = 3)
        else:
          ax.plot(mc_time,traj['params'][:,param_i],color=(0.6,0.6,0.6),alpha=0.8,zorder=1,linewidth = 3)

      #Plot mean MC traj
      ax.plot(mc_time,mean_param[:,param_i],color=(0,0.8,0),zorder=2,label='<evolution>',linewidth = 4)
      ax.tick_params(axis='both', which='both', labelsize=20)
      ax.set_xlabel('$\it{t}$',fontsize=34)
      ax.set_ylabel('$\it{x_i}$',fontsize=34,labelpad=20)
      ax.tick_params(which='major',direction='in',width=4,length=10,labelsize=30,pad = 10)
      ax.tick_params(which='minor',direction='in',width=4,length=0,labelsize=30)
      ax.legend(fontsize=24,fancybox=False,edgecolor=(0,0,0),framealpha=1,loc='upper right')
      fig.tight_layout()
      fig.savefig("Plots/Param_evo_%g.png"%(param_i),dpi=300)
  print("\nFinished")