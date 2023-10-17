import torch
from torch import nn
import numpy as np
from torchvision import models, datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt

retrain_model = False

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


model = models.alexnet(weights=models.AlexNet_Weights)#pretrained=True)


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

ndata = train_data.targets.shape[0]
ntrain = int(round(0.9 * ndata, 0))
nval = ndata - ntrain
train_subset, val_subset = torch.utils.data.random_split(train_data, [ntrain, nval])
#val_data, val_labels = train_data.data[val_subset.indices], train_data.targets[val_subset.indices]

batch_size = 4
trainloader = torch.utils.data.DataLoader(dataset=train_subset,
                                         batch_size=batch_size,
                                         shuffle=True
                                         )
valloader = torch.utils.data.DataLoader(dataset=val_subset,
                                         batch_size=batch_size,
                                         shuffle=False
                                         )

dataloader = torch.utils.data.DataLoader(train_data,
                                         batch_size=batch_size,
                                         shuffle=True
                                         )

criterion = nn.CrossEntropyLoss()
opt = optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)

if retrain_model:
    history = []
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data


            # reset gradient
            opt.zero_grad()
            probs = new_model(inputs)

            loss = criterion(probs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()

            if (i+1) % 1000 == 0: # every 100th minibatch
                valloss = 0
                model.eval()
                print("{0:d}, {1:5d} loss: {2:.3f}".format(epoch + 1, i + 1, running_loss/1000))
                # Validation loss
                for input, val_label in valloader:
                    y_hat = new_model(input)
                    valloss += criterion(y_hat, val_label)
                history.append([running_loss/1000, valloss.item() / len(valloader)])
                running_loss = 0.0
                model.train()

    torch.save(new_model, f"..\\..\\models\\alexnet_MNIST.pth")

    plt.plot(np.array([h[0] for h in history]), label='train loss')
    plt.plot(np.array([h[1] for h in history]), label='val loss')
    plt.legend()
    plt.show()

# ------------------------------------------------------------
# ----- model reloading and evaluation on test set -----------
# ------------------------------------------------------------
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# model = models.alexnet(weights=models.AlexNet_Weights)
model = torch.load(f"..\\..\\models\\alexnet_MNIST.pth")
model.eval() # set dropout and batchnorm layers to evaluation mode

test_data = datasets.MNIST(root="data",
                           train=False,
                           transform=transform)

data_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

# prediction phase
y_trues = []
y_preds = []
test_loss = 0

for i, data in enumerate(data_loader, 0):
    input, labels = data
    probs = model(input)
    y_trues.append(labels.detach().tolist())
    indices = np.argmax(probs.detach().tolist(), 1)
    y_preds.append(list(indices))
    test_loss += criterion(probs, labels)

y_trues = [i for sublist in y_trues for i in sublist]
y_preds = [i for sublist in y_preds for i in sublist]

print(f"loss: {test_loss/len(data_loader)}")
print(confusion_matrix(y_trues, y_preds))
print(f"accurace: {accuracy_score(y_trues, y_preds)}")