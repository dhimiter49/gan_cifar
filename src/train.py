import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tqdm

import nets

# specify absolute path


def train_dis(model, data_loader, device, optimizer, loss_function):
    model.train()
    train_loss = 0.0
    correct_pred = 0.0
    for (data, target) in tqdm(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        predictions = model(data)
        loss = loss_function(predictions, target)
        train_loss += loss.detach().item()
        loss.backward()
        optimize.step()

        # Logging
        predictions = (predictions >= 0.5)
        batch_correct_pred = predictions.eq(target).sum().item()
        correct_pred += batch_correct_pred

    return train_loss, correct_pred / len(data_loader.dataset)


def train_gen(gen_model, disc_model, latents, device, optimizer, loss_function):
    gen_model.train()
    disc_model.eval()
    train_loss = 0.0
    correct_pred = 0.0
    for data in tqdm(latents):
        data = data.to(device)

        optimizer.zero_grad()

        output = model(data)
        predictions = disc_model(ouput)
        loss = loss_function(predictions)
        train_loss += loss.detach().item()
        loss.backward()
        optimize.step()

        # Logging
        predictions = (predictions >= 0.5)
        batch_correct_pred = predictions.eq(target).sum().item()
        correct_pred += batch_correct_pred

    return train_loss, 1 - correct_pred / len(latents.flatten())


def test_dis(model, data_loader, device, loss):
    model.eval()
    test_loss = 0.0
    correct_pred = 0.0
    for (data, target) in tqdm(data_loader):
        data, target = data.to(device), target.to(device)


        predictions = model(data)
        loss = loss_function(predictions, target)
        test_loss += loss.detach().item()

        # Logging
        predictions = (predictions >= 0.5)
        batch_correct_pred = predictions.eq(target).sum().item()
        correct_pred += batch_correct_pred

    return test_loss, correct_pred / len(data_loader.dataset)


def test_gen(model, disc_moedl, latents, device, loss_function):
    model.eval()
    disc_model.eval()
    test_loss = 0.0
    correct_pred = 0.0
    for data in tqdm(latents):
        data = data.to(device)

        output = model(data)
        predictions = disc_model(ouput)
        loss = loss_function(predictions)
        test_loss += loss.detach().item()

        # Logging
        predictions = (predictions >= 0.5)
        batch_correct_pred = predictions.eq(target).sum().item()
        correct_pred += batch_correct_pred

    return test_loss, 1 - correct_pred / len(latents.flatten())


def main():
    # read config file
    # load dataset
    # instantiate dataloader
    # instantiate network, optimizer, loss...
    # main loop, calls train and test
    # log results
    # plot stuff
    pass

if __name__ == "__main__":
    main()

