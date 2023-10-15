import torch
from torch import nn
import numpy as np
from torchvision import models, datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt

class TransferFeatures(nn.Module):
    def __init__(self, original_model, classifier, modelname):
        '''
        extends originial model by custom leaf classifier
        :param original_model:
        :param classifier:  [N, K] -> Nclasses , where K is flattened dimension of originial_model output
        :param modelname:
        '''
        super(TransferFeatures, self).__init__()
        self.features = original_model.features
        self.classifier = classifier
        self.modelname = modelname

        # freeze weights of transfered model
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        # flatten output size: [N, K1, K2, ...] >> [N, K1*K2*...]
        f = f.view(f.size(0), np.prod(f.shape[1:]))
        probs = self.classifier(f)
        return probs


model = models.alexnet(pretrained=True)

classifier = nn.Sequential(nn.Dropout(),
                           nn.Linear(256 * 6 * 6, 4096),
                           nn.ReLU(inplace=True),
                           nn.Dropout(),
                           nn.Linear(4096, 4096),
                           nn.ReLU(inplace=True),
                           nn.Linear(4096, 10)
                           )

new_model = TransferFeatures(model, classifier, 'transfer')

# tranform images to size expected by alexnet
def transform(img):
    img = img.resize((224, 224)).convert("RGB")
    img = ToTensor()(img)
    return img


train_data = datasets.MNIST(root = "data",
                            train = True,
                            transform = transform,
                            download=True)


batch_size = 4
dataloader = torch.utils.data.DataLoader(train_data,
                                         batch_size=batch_size,
                                         shuffle=True
                                         )

criterion = nn.CrossEntropyLoss()
opt = optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)

history = []
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data


        # reset gradient
        opt.zero_grad()
        probs = new_model(inputs)

        loss = criterion(probs, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item()

        if i % 100 == 0: # every 100th minibatch
            print("{0:d}, {1:5d} loss: {2:.3f}".format(epoch + 1, i + 1, running_loss/100))
            history.append(running_loss/100)
            running_loss = 0.0

torch.save(new_model, "..\..\models\alexnet_MNIST.pth")

plt.plot(np.array(history))
plt.show()

# ------------------------------------------------------------
# ----- model reloading and evaluation on test set -----------
# ------------------------------------------------------------
from sklearn.metrics import confusion_matrix, classification_report

model.load(torch.load("..\..\models\alexnet_MNIST.pth"))
model.eval() # set dropout and batchnorm layers to evaluation mode

test_data = datasets.MNIST(root="data",
                           train=False,
                           transform=transform)

data_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1)

# prediction phase
y_trues = []
y_preds = []

for i, data in enumerate(data_loader, 0):
    input, labels = data
    output = model(input)
    y_trues.append(int(labels[0]))
    index = np.argmax(output)
    y_preds.append(int(index[0]))

print(confusion_matrix(y_trues, y_preds))