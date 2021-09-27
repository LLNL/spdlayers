import numpy as np
import torch
import torch.nn as nn
import spdlayers

n_epochs = 100
hidden_size = 100
out_shape = 6

X = np.loadtxt('input_Isotruss_solid.txt')
Y = np.loadtxt('C_Isotruss_solid.txt')
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

n_features = X.shape[1]
# create our 100x6x6 training tensors
Y = Y.view(-1, out_shape, out_shape)

# because we know that Y is orthtropic, the input_shape is 9
in_shape = 9

# define the model using the Eigen SPD layer
model = nn.Sequential(
          nn.Linear(n_features, hidden_size),
          nn.Linear(hidden_size, in_shape),
          spdlayers.Eigen(output_shape=out_shape,
                          symmetry='orthotropic',
                          positive='Exp')
        )
model = model.double()

loss_fn = nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0)

for epoch in range(n_epochs):
    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, Y)
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    print(f'Epoch: {epoch:4d} Loss: {loss.item():.5f}')
