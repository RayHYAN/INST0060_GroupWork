import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim



# implementation of Three-Layer Perceptron (classifier) using pytorch
class BasicMLP(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, output_dim, nonlinearity=nn.ReLU()):
        # in the constructor we build all of the components used by the model
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # model consists of a linear function, followed by a non-linearity
        # then a second linear function then a logistic sigmoid
        # Some call this a 2 layer and some a 3 layer network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.nonlinearity = nonlinearity
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        # to forward evaluate our network we:
        # calculate the activations at the hidden layer
        a = self.fc1(X)
        # then the hidden layer itself
        z = self.nonlinearity(a)
        # then the output activations
        output = self.fc2(z)
        # then the output (here between 0 and 1)
        output = self.sigmoid(output)
        return output
    
    def predict(self, X):
        # the prediction function outputs 0s or 1s from the soft predictions
        input_tensor = torch.tensor(X, dtype=torch.float)
        return np.array(
            [0 if i < 0.5 else 1 \
                for i in self(input_tensor).clone().detach().numpy()])
    
    def train(
            self, X, y, epochs=20000, optim_func=optim.SGD,
            loss_func=nn.BCELoss(), learning_rate=0.01, reg_lambda=0.01,
            print_loss=False, interactive_plots=True):
        """
        Train the network for multiple epochs.

        parameters
        ----------
        X - data matrix (a numpy 2d array )
        y - target vector (a numpy 1d array)
        epochs - number of epochs to train for
        optim_func (optional) - the pytorch optimiser to use, 
            default is Stochastic Gradient Descent
        loss_func (optional) - the pytorch loss function to use, 
            default is Binary Cross Entropy
        learning_rate (optional) - the scalar learning rate
        reg_lambda (optional) - the scalar regularisation parameter (unused)
        print_loss - set to true to output loss to screen after batch of epochs
        interactive_plots - set to true to see the decision boundary evolve
            during training
        """
        # if interactive plots then we want to set up the figure
        if interactive_plots:
            self.initialise_plots(X, y)
        # output layer assumes vector of outputs. If single target per datapoint
        # we must reshape the 1d array to be 2d.
        if len(y.shape) == 1:
            y = y.reshape((-1,1))
        # data matrix X and targets y must be converted to torch data structures
        in_tensor = torch.tensor(X, dtype=torch.float)
        target_tensor = torch.tensor(y, dtype=torch.float)
        print('Input shape', in_tensor.size())
        print('Target shape', target_tensor.size())
        # Initialise the optimiser
        optimiser = optim_func(self.parameters(), lr=learning_rate)
        #
        # training: each epoch evaluates the full gradient on X then takes a
        # step in that direction        
        for epoch in range(epochs):
            # Essential to restart the gradients before every epoch
            optimiser.zero_grad()
            # define the loss for this epoch
            loss = loss_func(self(in_tensor), target_tensor)
            # Perform backpropagation
            loss.backward()
            # Perform optimisation step
            optimiser.step()
            # output loss and/or update plot
            self.update_output(
                epoch, loss, print_loss, interactive_plots, in_tensor,
                target_tensor)

    def initialise_plots(self, X, y):
        # helper function for train that sets up interactive plot
        self.fig_ax = plt.subplots()
        fig, ax = self.fig_ax
        ax.scatter(
            X[:, 0], X[:, 1], s=40, c=y.ravel(), cmap=plt.cm.Spectral)

    def update_output(
            self, epoch, loss, print_loss, interactive_plots, in_tensor,
            target_tensor):
        # helper function to output loss and/or update plot as needed
        if epoch % 1000 != 0:
            return
        if print_loss:
            print("Loss after iteration %i: %f" % (epoch, loss))
        if interactive_plots:
            with torch.no_grad():
                plot_decision_boundary(
                    lambda x: self.predict(x), in_tensor, target_tensor,
                    fig_ax=self.fig_ax)

# implementation of Multi-Layer Perceptron using pytorch
class MLP(BasicMLP):
    def __init__(self, layer_dims, nonlinearity=nn.ReLU()):
        super(MLP, self).__init__(
            layer_dims[0],layer_dims[1],layer_dims[-1],
            nonlinearity=nonlinearity)
        self.layer_dims = layer_dims
        self.hidden = nn.ModuleList()
        # add the first linear layer initialised within BasicMLP.__init__
        self.hidden.append(self.fc1)
        # add the intermediate layers not initialised within BasicMLP.__init__
        for k in range(1, len(self.layer_dims)-2):
            self.hidden.append(nn.Linear(layer_dims[k], layer_dims[k+1]))
        # add the last linear layer initialised within BasicMLP.__init__
        self.hidden.append(self.fc2)
    
    def forward(self, X):
        # forward evaluation now involves multiple hidden layers
        output = X
        for h in range(len(self.hidden)):
            output = self.hidden[h](output)
            if h < len(self.hidden)-1:
                output = self.nonlinearity(output)
            else:
                output = self.sigmoid(output)
        return output

# Finally we put a plot function within the file so we can observe the 
# evolution of the decision boundary. Needs plt.ion() in main module
def plot_decision_boundary(pred_func, X, y, flush=True, fig_ax=None):
    if fig_ax is None:
        raise ValueError("This should not happen")
    else:
        fig, ax = fig_ax
    # min and max values (with padding)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # generate grid of points with distance h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # predict function value for whole grid    
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot contour and training examples
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    ax.scatter(X[:, 0], X[:, 1], marker='.', c=y.flatten(), cmap=plt.cm.Spectral)
    #
    fig.canvas.draw()
    if flush: fig.canvas.flush_events()

