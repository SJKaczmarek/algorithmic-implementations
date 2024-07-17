import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=False)
        self.r = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, h, c, n, m):
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h), dim=1)

        # Apply linear transformations
        gates = self.W(combined) + self.r(h) + self.b

        # Split into individual gates
        i, f, z, o = torch.split(gates, self.hidden_size, dim=1)

        # Exponential gating and stabilization
        i = torch.exp(i)
        f = torch.sigmoid(f)
        z = torch.tanh(z)
        o = torch.sigmoid(o)

        # Memory and normalization updates
        m = torch.max(torch.log(f) + m, torch.log(i))
        i_prime = torch.exp(torch.log(i) - m)
        f_prime = torch.exp(torch.log(f) + m - m)

        # Cell state and normalized state updates
        c = f_prime * c + i_prime * z
        n = f_prime * n + i_prime

        # Hidden state update
        h_tilde = c / n
        h = o * torch.tanh(h_tilde)

        return h, c, n, m

class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wq = nn.Linear(input_size, hidden_size, bias=False)
        self.Wk = nn.Linear(input_size, hidden_size, bias=False)
        self.Wv = nn.Linear(input_size, hidden_size, bias=False)
        self.Wi = nn.Linear(input_size, hidden_size, bias=False)
        self.Wf = nn.Linear(input_size, hidden_size, bias=False)
        self.Wo = nn.Linear(input_size, hidden_size, bias=False)

        self.bq = nn.Parameter(torch.zeros(hidden_size))
        self.bk = nn.Parameter(torch.zeros(hidden_size))
        self.bv = nn.Parameter(torch.zeros(hidden_size))
        self.bi = nn.Parameter(torch.zeros(hidden_size))
        self.bf = nn.Parameter(torch.zeros(hidden_size))
        self.bo = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, C, n):
        # Query, key, and value calculations
        q = self.Wq(x) + self.bq
        k = (1 / torch.sqrt(torch.tensor(self.hidden_size))) * (self.Wk(x) + self.bk)
        v = self.Wv(x) + self.bv

        # Input, forget, and output gate calculations
        i = torch.exp(self.Wi(x) + self.bi)
        f = torch.sigmoid(self.Wf(x) + self.bf)
        o = torch.sigmoid(self.Wo(x) + self.bo)

        # Memory and normalization updates
        C = f * C + i * torch.outer(v, k)
        n = f * n + i * k

        # Hidden state update
        h_tilde = torch.matmul(C, q) / torch.max(torch.matmul(n.transpose(0, 1), q), torch.tensor(1.0))
        h = o * torch.tanh(h_tilde)

        return h, C, n

class xLSTMTime(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, use_mlstm=False):
        super(xLSTMTime, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.use_mlstm = use_mlstm

        # Series Decomposition
        self.trend_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1)
        self.trend_pool = nn.AvgPool1d(kernel_size=3, padding=1)

        # Linear Transformation
        self.input_linear = nn.Linear(input_size, hidden_size)

        # xLSTM Block
        if self.use_mlstm:
            self.xlstm_cell = mLSTMCell(hidden_size, hidden_size)
        else:
            self.xlstm_cell = sLSTMCell(hidden_size, hidden_size)

        # Linear Transformation and Instance Normalization
        self.output_linear = nn.Linear(hidden_size, output_size)
        self.instance_norm = nn.InstanceNorm1d(output_size)

    def forward(self, x):
        # Series Decomposition
        trend = self.trend_pool(self.trend_conv(x))
        seasonal = x - trend

        # Linear Transformation and Batch Normalization
        x = self.input_linear(x)
        x = torch.nn.functional.batch_norm(x)

        # Initialize hidden state, cell state, normalization state, and memory
        if self.use_mlstm:
            h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
            C = torch.zeros(self.hidden_size, self.hidden_size).to(x.device)
            n = torch.zeros(x.size(0), self.hidden_size).to(x.device)
            m = None
        else:
            h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
            c = torch.zeros(x.size(0), self.hidden_size).to(x.device)
            n = torch.zeros(x.size(0), self.hidden_size).to(x.device)
            m = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        # xLSTM Block
        for i in range(self.seq_len):
            if self.use_mlstm:
                h, C, n = self.xlstm_cell(x[:, i, :], C, n)
            else:
                h, c, n, m = self.xlstm_cell(x[:, i, :], h, c, n, m)

        # Linear Transformation and Instance Normalization
        output = self.output_linear(h)
        output = self.instance_norm(output.unsqueeze(1)).squeeze(1)

        return output

# Example usage
input_size = 7  # Number of features
hidden_size = 64  # Hidden state size
output_size = 1  # Number of output values
seq_len = 96  # Sequence length
use_mlstm = False  # Use mLSTM or sLSTM

model = xLSTMTime(input_size, hidden_size, output_size, seq_len, use_mlstm)

# Input data (batch_size, seq_len, input_size)
x = torch.randn(32, seq_len, input_size)

# Forward pass
output = model(x)

# Print output shape
print(output.shape)
