import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR = args.hidRNN;
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.hidPhys = args.hidPhys;
        self.Ck = args.CNN_kernel;
        self.skip = args.skip;
        # disabled skip connection for now, just run simple RNN
        self.skip = -float("inf")

        self.pt = (self.P - self.Ck)/self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)); # [100, 1, 6, 321]
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p = args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        elif (self.hidPhys > 0):
            self.linear1 = nn.Linear(self.hidR + self.hidPhys, self.m); 
        else:
            self.linear1 = nn.Linear(self.hidR, self.m); # [321, 100]
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
 
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, 1, self.P, self.m); # [128, 1, 168, 321]
        c = F.relu(self.conv1(c)); # [128, 100, 163, 1]
        c = self.dropout(c);
        c = torch.squeeze(c, 3); # [128, 100, 163]
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous(); # [163, 128, 100]
        _, r = self.GRU1(r); # [1, 128, 100]
        r = self.dropout(torch.squeeze(r,0));

        # Physical knowledge
        if (self.hidPhys > 0):
            physics = self.physical_knowledge(x); # [128, 3]
            r = torch.cat((r, physics), 1); # [128, 103]
        
        #skip-rnn
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
            # print(batch_size, self.hidC, self.pt, self.skip)
            # print(type(batch_size), type(self.hidC), type(self.pt), type(self.skip))
            # raise NameError("debug")
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);

        res = self.linear1(r); # [128, 321];
        
        #highway - give more weight to recent data
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1,self.m);
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res;
    
        
    def physical_knowledge(self, x):
        """ 
        Compute physical knowledge from the input data.
        Physical knowledge includes the trend, variance across time, variance across population
        Input:
            x: [batch_size, P, m]
        Output:
            return: [batch_size, 3]
        """
        # trend
        trend = torch.mean(x[:, -5:, :] - x[:, :5, :], 1);
        trend = torch.mean(trend, 1);
        # variance across time
        var_time = torch.var(x, 1);
        var_time = torch.mean(var_time, 1);
        # variance across population
        var_pop = torch.var(x, 2);
        var_pop = torch.mean(var_pop, 1);

        return torch.stack((trend, var_time, var_pop), 1);

        
