import minitorch
import datasets
import time
import matplotlib.pyplot as plt

PTS = 50
DATASET = datasets.Xor(PTS, vis=True)
HIDDEN = 10
RATE = 0.5


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, HIDDEN)
        self.layer2 = Linear(HIDDEN, HIDDEN)
        self.layer3 = Linear(HIDDEN, 1)

    def forward(self, x):
        # TODO: Implement for Task 2.5.
        raise NotImplementedError('Need to implement for Task 2.5')


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # TODO: Implement for Task 2.5.
        raise NotImplementedError('Need to implement for Task 2.5')


model = Network()
data = DATASET

X = minitorch.tensor_fromlist(data.X)
y = minitorch.tensor(data.y)

losses = []
for epoch in range(250):
    total_loss = 0.0
    correct = 0
    start = time.time()

    # Forward
    out = model.forward(X).view(data.N)

    prob = (out * y) + (out - 1.0) * (y - 1.0)
    for i, lab in enumerate(data.y):
        if lab == 1 and out[i] > 0.5:
            correct += 1
        if lab == 0 and out[i] < 0.5:
            correct += 1

    loss = -prob.log()
    (loss.sum().view(1)).backward()
    total_loss += loss[0]
    losses.append(total_loss)

    # Update
    for p in model.parameters():
        if p.value.grad is not None:
            p.update(p.value - RATE * (p.value.grad / float(data.N)))

    epoch_time = time.time() - start

    # Logging
    if epoch % 10 == 0:
        print(
            "Epoch ",
            epoch,
            " loss ",
            total_loss,
            "correct",
            correct,
            "time",
            epoch_time,
        )
        im = f"Epoch: {epoch}"
        data.graph(im, lambda x: model.forward(minitorch.tensor(x, (1, 2)))[0, 0])
        plt.plot(losses, c="blue")
        data.vis.matplot(plt, win="loss")
