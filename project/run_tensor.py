import minitorch
import datasets
import time

PTS = 250
DATASET = datasets.Xor(PTS)
HIDDEN = 10
RATE = 0.5


class Network(minitorch.Module):
    def __init__(self):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, HIDDEN)
        self.layer2 = Linear(HIDDEN, HIDDEN)
        self.layer3 = Linear(HIDDEN, 1)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = minitorch.Parameter(
            2 * (minitorch.rand((in_size, out_size)) - 0.5)
        )
        self.bias = minitorch.Parameter(2 * (minitorch.rand((out_size,)) - 0.5))
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch, in_size, 1)
        ).sum(1).view(batch, self.out_size) + self.bias.value.view(1, self.out_size)


model = Network()
data = DATASET

X = minitorch.tensor([v for x in data.X for v in x], (data.N, 2))
y = minitorch.tensor(data.y)

for epoch in range(250):
    total_loss = 0.0
    correct = 0
    start = time.time()
    out = model.forward(X).view(data.N)
    out.name_("out")

    loss = (out * y) + (out - 1.0) * (y - 1.0)
    for i, lab in enumerate(data.y):
        if lab == 1 and out[i] > 0.5:
            correct += 1
        if lab == 0 and out[i] < 0.5:
            correct += 1

    # if epoch == 0:
    #     (-loss.log().sum().view(1)).make_graph("graph.dot", minitorch.tensor([1.0]))
    start = time.time()

    (-loss.log().sum().view(1)).backward()
    total_loss += loss[0]

    start = time.time()
    if epoch % 10 == 0:
        print("Epoch ", epoch, " loss ", total_loss, "correct", correct)
        im = "epoch%d.png" % epoch
        data.graph(im, lambda x: model.forward(minitorch.tensor(x, (1, 2)))[0, 0])

    for p in model.parameters():
        if p.value.grad is not None:
            p.update(p.value - 0.5 * (p.value.grad / float(data.N)))
