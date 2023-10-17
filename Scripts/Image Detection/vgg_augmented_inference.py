import numpy as np
import torch.cuda
from torchvision import transforms, models, datasets
import torch.nn as nn
from torch.optim import Adam
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4


def augment_image(img):
    # induce small shifts and rotations to mimic brain behaviour
    # where object is scanned by minor eye movements creating an ensemble of object embeddings
    pass

# read_images
def read_images(path):
    # read in and process all jpg files
    files = os.listdir(path)
    jpg_files = [file for file in files if file[-4:] == '.jpg']
    images = []
    for file in jpg_files[:1000]:
        try:
            image = Image.open(os.path.join(path, file))
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            image = image.convert(mode="RGB")
            image = np.asarray(image)
            images.append(image)
        except OSError:
            pass
    return images


def evaluate_model(model, dataloader, criterion):
    score = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        probs = model(inputs)
        score += criterion(probs, inputs)
    return score / len(dataloader)


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
        f = torch.flatten(f, 1)
        probs = self.classifier(f)
        return probs

# get vgg model
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

classifier = nn.Sequential(nn.Dropout(),
                           nn.Linear(512*7*7, 4096),
                           nn.ReLU(inplace=True),
                           nn.Dropout(),
                           nn.Linear(4096, 4096),
                           nn.ReLU(inplace=True),
                           nn.Linear(4096, 10)
                           )

model = TransferFeatures(vgg, classifier, "vgg_transfer")

print(model)


# Get data


def get_dataloders():
    train_data = datasets.MNIST(root = "data",
                                train = True,
                                transform = transform,
                                download=True)

    train_data = datasets.MNIST(root = "data",
                                train = True,
                                transform = transform,
                                download=True)

    ndata = train_data.targets.shape[0]
    ntrain = int(round(0.9 * ndata, 0))
    nval = ndata - ntrain
    train_subset, val_subset = torch.utils.data.random_split(train_data, [ntrain, nval])


    test_data = datasets.MNIST(root="data",
                               train=False,
                               transform=transform)

    dataloader_train = torch.utils.data.DataLoader(dataset= train_subset,
                                                   batch_szie = batch_size,
                                                   shuffle = True)

    dataloader_val = torch.utils.data.DataLoader(dataset= val_subset,
                                                 batch_size=batch_size,
                                                 shuffle=False)

    dataloader_test = torch.utils.data.DataLoader(dataset= test_data,
                                                   batch_szie = batch_size,
                                                   shuffle = True)
    return dataloader_train, dataloader_val, dataloader_test

dataloader_train, dataloader_val, dataloader_test = get_dataloders

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# training
criterion = nn.CrossEntropyLoss
opt = Adam(params=model.parameters(), lr=0.001)

for epoch in range(2):
    running_loss = 0
    for i, data in enumerate(dataloader_train, 0):
        inputs, labels = data

        opt.zero_grad()
        probs = model(inputs)
        loss = criterion(probs, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        if (i+1) % 1000 == 0:
            print(f"epoch {epoch+1:d}, step {i+1:5d}, loss {running_loss/1000:.3f}")
            running_loss = 0



acc = evaluate_model(model, dataloader_test)