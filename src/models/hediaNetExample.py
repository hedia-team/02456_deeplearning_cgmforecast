import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor


class DilatedNet(nn.Module):
    def __init__(self, n_steps_past=16, num_inputs=4,
                                     dilations=[1,1,2,4,8],
                                     h1=2,
                                     h2=3): #[32,32,32,64,64]):
        
        """
        :param num_inputs: int, number of input variables
        :param h1: int, size of first three hidden layers
        :param h2: int, size of last two hidden layers
        :param dilations: int, dilation value
        :param hidden_units:
        """
        
        super(DilatedNet, self).__init__()
        self.file_name = os.path.basename(__file__)
        self.hidden_units = [h1,h1,h1,h2,h2]
        self.dilations = dilations

        self.num_inputs = num_inputs
        self.receptive_field = sum(dilations) + 1

        self.input_width = n_steps_past # n steps past

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(self.num_inputs, self.hidden_units[0], kernel_size=2, dilation=self.dilations[0])
        self.conv2 = nn.Conv1d(self.hidden_units[0], self.hidden_units[1], kernel_size=2, dilation=self.dilations[1])
        self.conv3 = nn.Conv1d(self.hidden_units[1], self.hidden_units[2], kernel_size=2, dilation=self.dilations[2])
        self.conv4 = nn.Conv1d(self.hidden_units[2], self.hidden_units[3], kernel_size=2, dilation=self.dilations[3])
        self.conv5 = nn.Conv1d(self.hidden_units[3], 1, kernel_size=2, dilation=self.dilations[4])

       
    def forward(self, x):
        """
        :param x: Pytorch Variable, batch_size x n_stocks x T
        :return:
        """

        # First layer
        current_width = x.shape[2]
        pad = max(self.receptive_field - current_width, 0)
        input_pad = nn.functional.pad(x, [pad, 0], "constant", 0)

        x = self.relu(self.conv1(input_pad))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        # No relu on the last layer
        x = self.conv5(x)

        # Remove redundant dimensions
        out_final = x[:,:,-1]


        

        return out_final
