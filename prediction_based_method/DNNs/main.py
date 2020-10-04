import numpy as np
import os
import datetime
from shutil import copyfile

# load own modules
import net
import divergence
import visualization
import dataloader
import utils
import conf

# load PyTorch modules
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import argparse


# for reproducibility
np.random.seed(conf.seed)
torch.manual_seed(conf.seed)
rng = np.random.RandomState(conf.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(conf.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# training subroutine based on the training data set


def train(device, epoch, files, transform, params, model, optimizer, lr_scheduler, div, myplotter, filename, mean, std):

    # define data set and associated loader
    loaded_set = dataloader.DatasetFKM(files, transform.get_dict(
        'train'), mean, std, conf.input_type)
    loader = data.DataLoader(loaded_set, **params)

    # set model to training mode
    model.train()

    epoch_loss = 0.0

    # load one batch of data
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # predict parameter on sample data
        preds = model(inputs)

        # evaluate loss
        loss = criterion(preds, labels)

        l2_term = torch.tensor([0.0]).float()

        for i in np.arange(start=0, stop=2*(len(conf.fcs)-1), step=2):
            params = torch.cat([x.view(-1) for x in model.fc_layers[i].parameters()])
            l2_term += conf.l2_lambda*torch.norm(params, 1)

        loss += l2_term[0]

        # backpropagation and parameter updates
        loss.backward()
        optimizer.step()

        # running loss update
        epoch_loss += loss*inputs.size(0)

    epoch_loss = epoch_loss/len(loaded_set)

    # update the lr scheduler based on the epoch loss
    lr_scheduler.step(epoch_loss)
    print('epoch {} Training Loss: {:.7f}'.format(epoch, epoch_loss))

    # save to log file
    with open(foldername + '/logs/logs.txt', "a") as txt_file:
        txt_file.write('{}, {:.7f}, '.format(epoch, epoch_loss))

    # update plot of loss curve
    myplotter.loss_data(epoch, epoch_loss.detach(), 'train')

# training subroutine based on the test data set
@torch.no_grad()
def test(device, epoch, files, transform, params, model, div, myplotter, filename, mean, std, losses, optimizer):

    # define data set and associated loader
    loaded_set = dataloader.DatasetFKM(files, transform.get_dict(
        'test'), mean, std, conf.input_type)
    loader = data.DataLoader(loaded_set, **params)

    # set model to training mode
    model.eval()

    epoch_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # predict parameter on sample data
        preds = model(inputs)

        # evaluate loss
        loss = criterion(preds, labels)

        l2_term = torch.tensor([0.0]).float()

        for i in np.arange(start=0, stop=2*(len(conf.fcs)-1), step=2):
            params = torch.cat([x.view(-1) for x in model.fc_layers[i].parameters()])
            l2_term += conf.l2_lambda*torch.norm(params, 1)

        loss += l2_term[0]

        # running loss update
        epoch_loss += loss*inputs.size(0)

        with torch.no_grad():

            # push true and predicted parameter values to calculate the average predictions
            div.push_data(labels, preds)

    epoch_loss = epoch_loss/len(loaded_set)
    losses.append(epoch_loss)
    print('epoch {} Test Loss: {:.7f}'.format(epoch, epoch_loss))

    # save to log file
    with open(foldername + '/logs/logs.txt', "a") as txt_file:
        txt_file.write('{:.7f}, '.format(epoch_loss))

    # obtain averaged predictions
    p0, ppred = div.averaging()

    # calculate divergence based on predictions
    if conf.dim_parameter_space == 1:
        p0, ppred, dp, divp = div.divergence1d(p0, ppred)
    elif conf.dim_parameter_space == 2:
        p0, ppred, dp, divp = div.divergence2d(p0, ppred)
    else:
        print('Parameter space dimension is not supported!')

    # update the predicted phase diagram
    p0_save, ppred_save, dp_save, divp_save = myplotter.pltdiv(p0, ppred, dp, divp)

    # update plot of loss curve
    myplotter.loss_data(epoch, epoch_loss.detach(), 'test')
    myplotter._draw(0.1)

    # save parameter, predictions, and divergence for the given epoch
    div_dict = {'p0': p0_save.reshape((-1, conf.dim_parameter_space)), 'ppred': ppred_save.reshape((-1, conf.dim_parameter_space)),
                'dp': dp_save.reshape((-1, conf.dim_parameter_space)), 'divp': divp_save.reshape((-1, 1))}

    for key in div_dict:
        divname = foldername + '/div/{}_epoch{}.txt'.format(key, epoch)
        np.savetxt(divname, div_dict[key])

    # check if a stopping criterion is satisfied
    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
    print('learning rate: {}'.format(curr_lr))
    print('threshold: {}'.format(1e-8/(1-conf.lr_scheduler_factor)))

    if curr_lr <= 1e-8/(1-conf.lr_scheduler_factor):
        breaking = True
    else:
        breaking = False

    if epoch in conf.saving_epochs or breaking:
        # save the plotted predicted phase diagram for this epoch

        figname = foldername + '/figures/epoch{}.pdf'.format(epoch)
        myplotter.fig.savefig(figname, dpi=None, facecolor='w', edgecolor='w',
                              orientation='portrait', format=None,
                              transparent=False, bbox_inches=None, pad_inches=0.1,
                              metadata=None)

    return losses, breaking


if __name__ == '__main__':

    # set time
    dt = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)

    # choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data directory
    data_dir = conf.data_dir

    # create folder structure
    if conf.dim_parameter_space == 1:
        foldername = 'results/' + dt.strftime("%H-%M_%d-%m-%Y") + '_L{}_dim_{}_nftar_{}_input_{}_Umin_{}_Umax_{}_lr_{}_sched_{}_{}_l2_{}'.format(
            conf.dim, conf.dim_parameter_space, conf.nf_tar, conf.input_type, conf.U_min, conf.U_max, conf.lr, conf.lr_scheduler_factor, conf.lr_scheduler_patience, conf.l2_lambda)
    else:
        foldername = 'results/' + dt.strftime("%H-%M_%d-%m-%Y") + '_L{}_dim_{}_input_{}_lr_{}_sched_{}_{}_l2_{}'.format(
            conf.dim, conf.dim_parameter_space, conf.input_type, conf.lr, conf.lr_scheduler_factor, conf.lr_scheduler_patience, conf.l2_lambda)

    print(foldername)

    for name in ['', '/logs', '/figures', '/div', '/trained_model', '/code', '/final_figures']:
        os.makedirs(foldername + name)

    with open(foldername + '/logs/logs.txt', "w") as txt_file:
        txt_file.write(
            'epoch, epoch_loss_training, epoch_loss_test, epoch duration (s)' + '\n')

    for name in ['dataloader', 'divergence', 'conf', 'main', 'net', 'utils', 'visualization', 'plotting']:
        copyfile('./{}.py'.format(name), foldername +
                 '/code/{}.py'.format(name))

    # initialize model
    model = net.Net().to(device)

    print(model)
    print(sum([np.prod(p.size()) for p in model.parameters()]))

    # split data set into training and test set
    trainset, testset = utils.construct_partitions_FKM(data_dir)

    # Parameters for dataloader
    params = conf.params
    params_stand = conf.params_stand

    # Initialize data loader
    if conf.input_type == 'corr_func':

        num_inputs = 3*int(round(conf.dim/2))

        loaded_set = dataloader.DatasetFKM(trainset, None, torch.zeros(
            [1, num_inputs]), torch.ones([1, num_inputs]), conf.input_type)
    else:
        loaded_set = dataloader.DatasetFKM(trainset, None, torch.zeros(
            [1, conf.dim, conf.dim]), torch.ones([1, conf.dim, conf.dim]), conf.input_type)
    loader = data.DataLoader(loaded_set, **params_stand)

    # Obtain standardization
    if conf.input_type == 'corr_func':

        num_inputs = 3*int(round(conf.dim/2))

        standardization_stats = []
        for i in range(num_inputs):
            standardization_stats.append(utils.RunningStats())

        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            for i in range(num_inputs):
                standardization_stats[i].push(torch.flatten(inputs)[i])

        mean_np = np.zeros(num_inputs)
        std_np = np.zeros(num_inputs)

        for i in range(num_inputs):
            mean_np[i] = standardization_stats[i].mean()
            std_np[i] = standardization_stats[i].standard_deviation()
            if std_np[i] == 0:
                std_np[i] = 1

        mean = torch.from_numpy(mean_np).view(1, num_inputs).float()
        std = torch.from_numpy(std_np).view(1, num_inputs).float()

        np.savetxt(foldername + '/trained_model/mean.txt',
                   mean.squeeze().detach().numpy())
        np.savetxt(foldername + '/trained_model/std.txt',
                   std.squeeze().detach().numpy())

    else:
        standardization_stats = []
        for i in range(conf.dim**2):
            standardization_stats.append(utils.RunningStats())

        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            for i in range(conf.dim**2):
                standardization_stats[i].push(torch.flatten(inputs)[i])

        mean_np = np.zeros(conf.dim**2)
        std_np = np.zeros(conf.dim**2)

        for i in range(conf.dim**2):
            mean_np[i] = standardization_stats[i].mean()
            std_np[i] = standardization_stats[i].standard_deviation()
            if std_np[i] == 0:
                std_np[i] = 1

        mean = torch.from_numpy(mean_np).view(1, conf.dim, conf.dim).float()
        std = torch.from_numpy(std_np).view(1, conf.dim, conf.dim).float()

        np.savetxt(foldername + '/trained_model/mean.txt',
                   mean.squeeze().detach().numpy())
        np.savetxt(foldername + '/trained_model/std.txt',
                   std.squeeze().detach().numpy())

    # setup transformations for online data augmentation
    transform = dataloader.Transformations()

    # set model to training mode
    model.train()

    epoch_loss = 0.0

    # define loss function (MSE loss)
    criterion = nn.MSELoss()

    # define optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)

    # set LR scheduler (ReduceLROnPlateau)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=conf.lr_scheduler_factor, patience=conf.lr_scheduler_patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.0, eps=1e-09)

    # class to cpmpute divergence
    div = divergence.vfdivergence()

    # to plot the progress
    myplotter = visualization.myplotter()

    # start training and evaluating
    losses = []
    for epoch in range(conf.epochs):
        with utils.Timer('epoch {} timing: '.format(epoch), foldername + '/logs/logs.txt'):
            train(device, epoch, trainset, transform, params,
                  model, optimizer, lr_scheduler, div, myplotter, foldername, mean, std)
            losses, breaking = test(device, epoch, testset, transform, params,
                                    model, div, myplotter, foldername, mean, std, losses, optimizer)

            # store final network state for serialization
            if (epoch in conf.saving_epochs) or breaking:
                torch.save({
                    'epoch': conf.epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, foldername + '/trained_model/model_epoch{}.pth'.format(epoch))

            # if stopping criterion is satisfied, end the training
            if breaking:
                myplotter.close()
                np.savetxt(foldername + '/logs/last_epoch.txt', [epoch])
                break

        print('--'*10)
