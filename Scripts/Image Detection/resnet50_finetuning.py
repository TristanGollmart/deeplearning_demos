# Performs transfer learning on the resnet50 model on mnist dataset
# Evaluates performance on a testset using standard and augmented inference, respectively

import numpy as np
import sklearn.metrics
import torch.cuda
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from torch.optim import Adam
import os
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
train_model = False

def augment_image(img):
    # induce small shifts and rotations to mimic brain behaviour
    # where object is scanned by minor eye movements creating an ensemble of object embeddings
    #orig_device = img.get_device()
    img = img.cpu()

    rotated = [transforms.RandomRotation(degrees=d)(img) for d in np.random.randint(0, 20, size=10)]
    cropped = [transforms.Resize(size=img.shape[-1])(transforms.CenterCrop(size=int(size))(img))
               for size in np.dot(np.shape(img)[-1], [0.7, 0.8, 0.9, 1.]).astype(int)]
    augmented_images = np.concatenate([rotated, cropped])
    #img = img.to(orig_device)
    return augmented_images


def evaluate_model(model, dataloader, criterion, augment_images=False, aggregation_augmentations=None,
                   aggregation_logits=None):
    """
    evaluates model on data yield by dataloader
    :param model: torch model to be evaluated
    :param dataloader: torch DataLoader structure for data to evaluate model on
    :param criterion: evaluation metric
    :param augment_images: [bool] if False, use standard inference, else augmented inference
    :param aggregation_augmentations: Way to aggregate the single predictions of the augmented inputs
    :param aggregation_logits: If metric expects single inputs, not lists:
            way to aggregate the vector of probabilities
    :return: returns average score
    """
    # model.to("cpu")

    if not aggregation_logits in ["max", None]:
        raise ValueError("aggregation of probability vector must be one of {0}".
                         format(["max", None]))

    model.eval()
    with torch.no_grad():
        score = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            if augment_images:
                if not aggregation_augmentations in ["sum", "avg", "max", None]:
                    raise ValueError("aggregation of augmented results must be one of {0}".
                                     format(['sum', 'avg', "max", None]))

                probs = []
                for ix, input in enumerate(inputs):
                    augmented_inputs = augment_image(input.type(torch.float32))
                    # label = labels[ix]
                    # f, ax = plt.subplots(1, len(augmented_inputs))
                    # for i, img in enumerate(augmented_inputs):
                    #     ax[i].imshow(img.permute(1, 2, 0))
                    # plt.show()

                    if aggregation_augmentations == "max":
                        probs.append(torch.max(torch.stack([model(single_input.unsqueeze(0).to(device)).detach()
                                                            for single_input in augmented_inputs]), axis=0).to(device))
                    elif aggregation_augmentations == "avg":
                        probs.append(torch.mean(torch.stack([model(single_input.unsqueeze(0).to(device)).detach()
                                                             for single_input in augmented_inputs]), axis=0).to(device))
                    elif aggregation_augmentations == None:
                        probs.append(torch.stack([model(single_input.unsqueeze(0).to(device)).detach()
                                                  for single_input in augmented_inputs]).to(device))

                # print(torch.stack(probs).get_device())
                # print(labels.get_device())
                if aggregation_logits == None:
                    score += criterion(torch.stack(probs).squeeze(1), labels)
                elif aggregation_logits == "max":
                    score += criterion(torch.argmax(torch.stack(probs).squeeze(1), dim=1), labels)

            else:
                probs = model(inputs.type(torch.float32))
                if aggregation_logits == None:
                    score += criterion(probs, labels)
                elif aggregation_logits == "max":
                    score += criterion(torch.argmax(probs, dim=1), labels)
    return score / len(dataloader)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return np.shape(self.y)[0]

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

# get resnet model, extract the feature layers, define new classifier head for custom problem
resnet_weights = models.ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=resnet_weights).to(device)

for parameter in resnet.parameters():
    parameter.requires_grad = False

classifier = nn.Sequential(nn.Linear(1000, 512),
                           nn.ReLU(inplace=True),
                           nn.Linear(512, 10)
                           )

print(list(resnet.modules()))
print(classifier)

feature_model = resnet
model = nn.Sequential(feature_model, classifier)
model.to(device)

# Get data
transform_train = transforms.Compose([
        #im.convert("RGB")
        #transforms.ToTensor(),
        transforms.Lambda(lambda x: x if torch.is_tensor(x) else transforms.ToTensor()(x)),
        #transforms.Lambda(lambda x: x.unsqueeze_(1).repeat(1, 3, 1, 1)),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1) if len(x.shape) <= 2 else x.unsqueeze(1).repeat(1, 3, 1, 1)),
        transforms.Lambda(lambda x: x.unsqueeze(0).expand(3, -1, -1) if len(x.shape) <= 2 else x.expand(3, -1, -1)), #batches are created automatically by dataloader: 3, N, N -> B, 3, N, N
        transforms.RandomResizedCrop(size=(224, 224), antialias=True),
        transforms.RandomRotation((-15, 15)),
        transforms.Resize((232, 232)),
        #transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x if torch.is_tensor(x) else transforms.ToTensor()(x)),
        #transforms.Lambda(lambda x: x.unsqueeze_(1).repeat(1, 3, 1, 1)),
        transforms.Lambda(lambda x: x.unsqueeze(0).expand(3, -1, -1) if len(x.shape) <= 2 else (x.expand(3, -1, -1) if len(x.shape) == 3 else x.expand(-1, 3, -1, -1))),
        transforms.Resize((232, 232)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

mnist = datasets.MNIST(root = "data",
                                train = True,
                                transform = transform_train,
                                download=True)

images = transform_train(mnist.data[:1].type(torch.float32))
plt.imshow(images.permute(1, 2, 0))
plt.imshow(mnist.data[0])

def get_dataloaders(transforms_train, transforms_test):
    """
    returns dataloaders for train, validation and test data
    :param feature_model: pretrained model to create useful features/ embeddings
    :param transforms_train: torch transforms to apply to training data
    :param transforms_test:  torch transforms to apply to test data
    :return:
    """

    transform_PIL_to_NP = transforms.Lambda(lambda x: np.array(x))
    data = datasets.MNIST(root = "data",
                                train = True,
                                transform = transforms.Compose([transform_PIL_to_NP, transform_train]),
                                download=True)

    # train_dataloader = DataLoader(dataset=data,
    #                             shuffle=True)


    ndata = len(data) #train_dataloader.dataset.targets.size()[0]
    ntrain = int(round(0.9 * ndata, 0))
    nval = ndata - ntrain

    train_subset, val_subset = torch.utils.data.random_split(data, [ntrain, nval])

    # Test Data
    test_data = datasets.MNIST(root="data",
                               train=False,
                               transform=transforms.Compose([transform_PIL_to_NP, transform_test]))
    test_data_subset = Subset(test_data, indices=range(len(test_data)))


    dataloader_train = DataLoader(dataset= train_subset,
                                                   batch_size = batch_size,
                                                   shuffle = True)
    dataloader_val = DataLoader(dataset= val_subset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    dataloader_test = DataLoader(dataset= test_data_subset,
                                                   batch_size = batch_size,
                                                   shuffle = True)

    return dataloader_train, dataloader_val, dataloader_test

dataloader_train, dataloader_val, dataloader_test = get_dataloaders(transform_train, transform_test)



# training classifier head on extracted features
criterion = nn.CrossEntropyLoss()

if train_model:
    opt = Adam(params=classifier.parameters(), lr=0.0001)
    history = []
    for epoch in range(1):
        running_loss = 0
        for i, data in enumerate(dataloader_train, 0):
            inputs, labels = data

            opt.zero_grad()
            probs = model(inputs)
            loss = criterion(probs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            history.append(loss.item())
            if ((i+1) * batch_size) % 1000 == 0:
                print(f"epoch {epoch+1:d}, step {i+1:5d}, loss {running_loss/ np.round(1000/batch_size):.3f}")
                running_loss = 0
                break

    plt.plot(history)
    plt.show()
    torch.save(model, f"..\\..\\models\\resnet50_finetuned_Mnist.pth")

model = torch.load(f"..\\..\\models\\resnet50_finetuned_Mnist.pth", map_location=torch.device('cpu'))
for i, data in enumerate(dataloader_test, 0):
    inputs, labels = data
    probs = model(inputs.type(torch.float32))
    print(probs[0])
    break

#loss = evaluate_model(model, dataloader_test, criterion)
#loss_train = evaluate_model(model, dataloader_train, criterion)
#print(f"loss without augmentation: {loss:.3f}")

#loss_augmented = evaluate_model(model, dataloader_test, criterion, True, aggregation_augmentations="avg")
#print(f"loss with augmentation: {loss_augmented:.3f}")

from torchmetrics import Accuracy
num_classes = len(set(dataloader_train.dataset.dataset.targets.tolist()))
Accuracy(task="multiclass", num_classes=num_classes)(torch.Tensor([1., 1., 1., 1.]).type(torch.int), torch.Tensor([1., 2., 3., 1.]).type(torch.int))
acc = evaluate_model(model, dataloader_test, Accuracy("multiclass", num_classes=num_classes), False, aggregation_logits="max")
acc_aug = evaluate_model(model, dataloader_test, Accuracy("multiclass", num_classes=num_classes), True, aggregation_augmentations="avg", aggregation_logits="max")

print(f"accuracy without augmentation: {acc:.3f}")
print(f"accuracy with augmentation: {acc_aug:.3f}")