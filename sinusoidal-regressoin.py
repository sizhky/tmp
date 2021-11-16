import pytorch_lightning as pl
from torch.utils.data import TensorDataset

from torch_snippets import *

x = np.linspace(-10, 10, 100)
y = np.sin(x)

plt.scatter(x, y)

x, y = [torch.Tensor(i) for i in [x,y]]
train_loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)

class Net(pl.LightningModule):
    loss = nn.L1Loss()
    def __init__(self):
        super().__init__()
        x = torch.ones(1)*0
        self.params = nn.Parameter(data=x)

    def forward(self, x):
        return torch.sin(x*self.params[0])

def plot(x, y, net):
    plt.scatter(x.cpu().detach().numpy(), net(x).cpu().detach().numpy(), label='pred')
    plt.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy(), label='truth')
    plt.legend()
    plt.show()
    
def train_batch(model, batch):
    x, y = batch
    optimizer.zero_grad()
    y_hat = model(x)
    loss = model.loss(y_hat, y)
    # print(y, y_hat)
    # print(loss)
    # plot(x, y, model)
    loss.backward()
    optimizer.step()
    return loss

e = 1
net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
r = Report(e)

for ex in range(100):
    for bx, b in enumerate(train_loader):
        loss = train_batch(net, b)
        r.record(
            (ex + (1+bx)/len(train_loader)),  
            param=net.params.item(),
            loss=loss, end='\r')
    if ex%10==0:
        r.report_avgs(ex+1)
        
