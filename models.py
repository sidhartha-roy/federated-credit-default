import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, accuracy_score
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(121, 240)
        self.fc2 = nn.Linear(240, 160)
        self.fc3 = nn.Linear(160, 80)
        self.fc4 = nn.Linear(80, 20)
        self.fc5 = nn.Linear(20, 2)

    def forward(self, x):
        x = x.view(-1, 121)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc5(x), dim=1)
        return x


def update_model(data, target, model, optimizer):
    target = target.long().view(-1)
    model.send(data.location)
    optimizer.zero_grad()
    prediction = model(data)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prediction, target)
    loss.backward()
    optimizer.step()
    return model


def test_model(federated_model, dataset):
    federated_model.eval()
    test_loss = 0
    avg_precision = 0
    avg_accuracy = 0
    avg_recall = 0
    num_iter = 0
    for data, target in dataset.test_loader:
        target = target.long().view(-1)
        output = federated_model.forward(data)
        criterion = nn.CrossEntropyLoss()
        test_loss += criterion(output, target).item()
        prediction = output.data.max(1, keepdim=True)[1]
        target_array = target.numpy().flatten()
        prediction_array = prediction.numpy().flatten()
        avg_precision += precision_score(target_array, prediction_array)
        avg_accuracy += accuracy_score(target_array, prediction_array)
        avg_recall += recall_score(target_array, prediction_array)
        num_iter += 1

    test_loss /= num_iter  # len(test_loader.dataset)
    avg_precision /= num_iter
    avg_accuracy /= num_iter
    avg_recall /= num_iter
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    print('Avg Accuarcy: {:.4f}'.format(avg_accuracy))
    print('Avg Precision: {:.4f}'.format(avg_precision))
    print('Avg Recall: {:.4f}'.format(avg_recall))

    return test_loss, avg_precision, avg_accuracy, avg_recall


def train_on_devices(remote_dataset, devices, models, optimizers):
    # iterate through each worker's dataset seperately
    for data_index in range(len(remote_dataset[0]) - 1):
        for device_index in range(len(devices)):
            data, target = remote_dataset[device_index][data_index]
            models[device_index] = update_model(data, target, models[device_index], optimizers[device_index])

        for model in models:
            model.get()

        return utils.federated_avg({
            'bob': models[0],
            'alice': models[1]
        })