# Performs transfer learning on the resnet50 model on mnist dataset
# Evaluates performance on a testset using standard and augmented inference, respectively

import numpy as np
import torch.cuda
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from torch.optim import Adam
import os
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4


def augment_image(img):
    # induce small shifts and rotations to mimic brain behaviour
    # where object is scanned by minor eye movements creating an ensemble of object embeddings
    rotated = [transforms.RandomRotation(degrees=d)(img) for d in np.random.randint(-20, 20, size=10)]
    resized = [transforms.Resize(sizr=size)(img) for size in np.shape(img)[0]*[0.7, 0.8, 0.9, 1.]]
    cropped = [transforms.CenterCrop(size=size)(img) for size in np.shape(img)*[0.7, 0.8, 0.9, 1.]]
    return rotated.append(resized.append(cropped))



def evaluate_model(model, dataloader, criterion, augment_images=False):
    model.eval()
    score = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        if augment_images:
            probs = []
            for ix, input in enumerate(inputs):
                augmented_inputs = augment_image(input)
                label = labels[ix]
                probs.append(np.average([model(single_input) for single_input in augmented_inputs]))
        else:
            probs = model(inputs)

        score += criterion(probs, inputs)
    return score / len(dataloader)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return np.shape(y)[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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

#model = TransferFeatures(vgg, classifier, "vgg_transfer")


print(vgg.features)
print(classifier)



# Get data
transform_train = transforms.Compose([
        #im.convert("RGB")
        #transforms.ToTensor(),
        transforms.Lambda(lambda x: x if torch.is_tensor(x) else transforms.ToTensor()(x)),
        #transforms.Lambda(lambda x: x.unsqueeze_(1).repeat(1, 3, 1, 1)),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1) if len(x.shape) <= 2 else x.unsqueeze(1).repeat(1, 3, 1, 1)),
        transforms.Lambda(lambda x: x.unsqueeze(0).expand(3, -1, -1) if len(x.shape) <= 2 else x.unsqueeze(1).expand(-1, 3, -1, -1)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x if torch.is_tensor(x) else transforms.ToTensor()(x)),
        #transforms.Lambda(lambda x: x.unsqueeze_(1).repeat(1, 3, 1, 1)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if len(x.shape()) == 3 else x.repeat(1, 3, 1, 1)),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

mnist = datasets.MNIST(root = "data",
                                train = True,
                                transform = transform_train,
                                download=True)

images = transform_train(mnist.data[1].type(torch.float32))

def get_dataloaders(feature_model, transforms_train, transforms_test):
    """
    returns dataloaders for train, validation and test data
    :param feature_model: pretrained model to create useful features/ embeddings
    :param transforms_train: torch transforms to apply to training data
    :param transforms_test:  torch transforms to apply to test data
    :return:
    """
    feature_model.to(device)
    transform_PIL_to_NP = transforms.Lambda(lambda x: np.array(x))
    train_data = datasets.MNIST(root = "data",
                                train = True,
                                transform = transform_PIL_to_NP,
                                download=True)

    train_data_subset = Subset(train_data, indices=range(10000))

    dataloader_temp = DataLoader(dataset=train_data_subset,
                                batch_size=1000,
                                shuffle=True)


    with torch.no_grad():
        x = []
        y = []
        for i, data in enumerate(dataloader_temp, 0):
            input, targets = data
            data = transforms_train(input.type(torch.float32))
            y.append(torch.flatten(targets).tolist())
            features = feature_model(data)
            features = torch.flatten(features, 1).tolist()
            x += features

    y = np.reshape(y, -1)
    y = torch.tensor(y)
    x = torch.tensor(x)
    dataset_train = CustomDataset(x, y)

    ndata = dataset_train.y.shape[0]
    ntrain = int(round(0.9 * ndata, 0))
    nval = ndata - ntrain
    train_subset, val_subset = torch.utils.data.random_split(dataset_train, [ntrain, nval])


    test_data = datasets.MNIST(root="data",
                               train=False,
                               transform=None)

    x = transforms_test(test_data.data)
    features = feature_model(x)
    features = torch.flatten(features, 1)
    dataset_test = CustomDataset(features, test_data.targets)

    dataloader_train = DataLoader(dataset= train_subset,
                                                   batch_size = batch_size,
                                                   shuffle = True)

    dataloader_val = DataLoader(dataset= val_subset,
                                                 batch_size=batch_size,
                                                 shuffle=False)

    dataloader_test = DataLoader(dataset= dataset_test,
                                                   batch_size = batch_size,
                                                   shuffle = True)
    return dataloader_train, dataloader_val, dataloader_test

dataloader_train, dataloader_val, dataloader_test = get_dataloaders(vgg.features, transform_train, transform_test)



# training
criterion = nn.CrossEntropyLoss()
opt = Adam(params=classifier.parameters(), lr=0.001)

history = []
for epoch in range(2):
    running_loss = 0
    for i, data in enumerate(dataloader_train, 0):
        inputs, labels = data

        opt.zero_grad()
        probs = classifier(inputs)
        loss = criterion(probs, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        history.append(loss.item())
        if (i+1) % 1000 == 0:
            print(f"epoch {epoch+1:d}, step {i+1:5d}, loss {running_loss/1000:.3f}")
            running_loss = 0

torch.save(model, f"..\\..\\models\\vgg16_transfer_augmented_inference.pth")

plt.plot(history)
plt.show()



acc = evaluate_model(classifier, dataloader_test, criterion)

acc_augment = evaluate_model(model, dataloader_test, criterion, True)

print(f"accuracy without augmentation: {acc:.3f}")
print(f"accuracy with augmentation: {acc_augment:.3f}")