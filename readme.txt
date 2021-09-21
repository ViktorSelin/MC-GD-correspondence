Code for the paper "Correspondence between neuroevolution and gradient descent"
Authors: Stephen Whitelam, Viktor Selin, Sang-Won Park, and Isaac Tamblyn

File Structure:
MC_GD correspondence files/
	|_ model.py
	|_ train_functions.py
	|_ readme.txt
	|_ Data/
		|_ sin(1pix).npz
	|_ Trajectories/
	|_ Plots

------------------------------------------------------
To run code
------------------------------------------------------
Do 

python train_functions.py

to do supervised learning of the function contained in the folder Data. The python script runs one trajectory of clipped gradient descent and a number (default=10) of trajectories of zero-temperature Monte Carlo (neuroevolution). 
It outputs several weights and the loss, and saves raw data files in the folder Trajectories, and .png images in the folder Plots. 

View comments in train_functions.py in order to modify the code.




------------------------------------------------------
File descriptions:
------------------------------------------------------
Data/: 	
	Folder containing the target function used in the paper
	Load it using:
	data = np.load('Data/sin(1pix).npz')

model.py:
	Contains the pytorch neural network class
	import the Net class and define a model as:
	model = Net(depth,width,input_size,output_size)

train_functions.py:
	Contains two functions, one to train using clipped gradient descent
	and one to train using zero-temperature MC

Trajectories/:
	Folder where trajectories are saved after running train_functions.py

Plots/:
	Folder where plots are saved after running train_functions.py


------------------------------------------------------
Python information:
------------------------------------------------------
Python: 3.7.12
Numpy: 1.19.5
torch: 1.9.0+cu102