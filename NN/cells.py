import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    Standard LSTM cell
    """
    def __init__(self, input_size : int, hidden_size : int):
        super().__init__()
        # set initial parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(input_size,4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size,4 * hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))
    @staticmethod
    def __str__():
        return "LSTM"
    def forward(self,x,hx,cx ):
        # forward method of LSTM
        # do gate matric operations
        gates = (torch.mm(x, self.weight_ih) + self.bias +
                 torch.mm(hx, self.weight_hh))
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # apply activation functions
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # update cell and hidden state
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy
    
class PeepholeLSTMCell(nn.Module):
    """
    Peephole LSTM
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # initialise parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(input_size,3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size,3 * hidden_size))
        self.bias =nn.Parameter(torch.randn(3*hidden_size))
        self.weight_c =nn.Parameter(torch.randn(input_size,hidden_size))
        self.bias_c =nn.Parameter(torch.randn(hidden_size))
    @staticmethod
    def __str__():
        return "PeepholeLSTM"
    def forward(self, x,hx,cx ):
        # forward of cell takes hidden state despite it not being used so it can be interchanged with standard LSTM
        # do matrix operations
        gates = (torch.mm(x, self.weight_ih) + self.bias +
                 torch.mm(cx, self.weight_hh) )
        ingate, forgetgate, outgate = gates.chunk(3, 1)
        
        # activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        
        outgate = torch.sigmoid(outgate)

        # get new cell and hidden state
        cy = (forgetgate * cx) + (ingate * torch.sigmoid(torch.mm(x, self.weight_c)+self.bias_c))
        hy = outgate*cy

        return hy,cy
class GRUCell(nn.Module):
    """
    Grated Recurrent Unit Cell
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(input_size,2 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size,2 * hidden_size))
        self.bias =nn.Parameter(torch.randn(2*hidden_size))
        self.weight_ic =nn.Parameter(torch.randn(input_size,hidden_size))
        self.weight_hc =nn.Parameter(torch.randn(hidden_size,hidden_size))
        self.bias_c =nn.Parameter(torch.randn(hidden_size))
    @staticmethod
    def __str__():
        return "GRU"
    def forward(self, x,hx,cx):
        gates = (torch.mm(x, self.weight_ih) + self.bias +
                 torch.mm(hx, self.weight_hh) )
        updategate, resetgate = gates.chunk(2, 1)

        updategate = torch.sigmoid(updategate)
        resetgate = torch.sigmoid(resetgate)
        
        hy = (updategate * hx) + (1.0-updategate)*torch.tanh(torch.mm(x,self.weight_ic)+torch.mm(resetgate*hx,self.weight_hc)+self.bias_c)
        
        return hy,cx

