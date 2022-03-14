import torch
import torch.nn as nn
import torch.nn.functional as F

CARTPOLE_N_ACTIONS=2
CARTPOLE_STATE_DIM=4

class CartPoleAgent(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(CARTPOLE_STATE_DIM, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, CARTPOLE_N_ACTIONS)
        )

    def forward(self, x):
        return self.network(x)
    
    def predict_probs(self, states):
        """ 
        Predict action probabilities given states.
        :param states: numpy array of shape [batch, state_shape]
        :returns: numpy array of shape [batch, n_actions]
        """
        input_states = torch.as_tensor(states, device='cuda', dtype=torch.float32)
        logits = self.network(input_states)
        probs = F.softmax(logits, dim=1)
        return probs.cpu().data.numpy()
