import numpy as np
from plotting import plot_linescan
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import conf

# set a random seed
seed = conf.seed
np.random.seed(seed)

# translate the input configuration


def roll(image):
    image_transf_1 = np.roll(image, np.random.randint(image.shape[0]), axis=0)
    image_transf_2 = np.roll(image_transf_1, np.random.randint(image.shape[0]), axis=1)
    return image_transf_2

# rotate the input configuration


def rot(image):
    image_transf = np.rot90(image, np.random.randint(4))
    return image_transf

# reflect the input configuration


def flip(image):
    rand_num = np.random.randint(3)
    if rand_num == 0 or rand_num == 1:
        return np.flip(image, axis=rand_num)
    else:
        return image

# process a given input configuration by applying a sequence of random symmetry transformations, thereby constructing a random transformation of p4m (plane symmetry group of square lattice)


def symmetrize(image, ft):
    if ft:
        which = {0: rot, 1: flip}

        sequence = np.random.permutation(2)

        for index in sequence:
            image = which[index](image)

        return image
    else:
        which = {0: roll, 1: rot, 2: flip}

        sequence = np.random.permutation(3)

        for index in sequence:
            image = which[index](image)

        return image

# calculate the magnitude of the discrete FT


def fouriertrafo(image):
    ft_image = np.fft.fft2(image)

    ft_image = np.fft.fftshift(ft_image)
    image_ft_magn = np.abs(ft_image)

    return image_ft_magn


# linear lattice size L
dim = 20

# range of U and Nf under investigation
U_min = conf.U_min
U_max = conf.U_max
U_step = conf.U_step
U_range = np.arange(U_min, U_max+U_step, U_step)
nf_range = range(1, int(round(dim**2/2+1)))

# set path to data folder and specify the range of 'versions', i.e., configurations which should be considered at each point in parameter space
if conf.case == 'noise-free':
    v_range = range(1)
    data_path = conf.path_data_noisefree
elif conf.case == 'noisy':
    v_range = range(1, 11)
    data_path = conf.path_data_noisy

# get the proper shape of input and output arrays by loading a first input
if conf.input_type == 'ft' or conf.input_type == 'raw':
    path = data_path + 'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_min, conf.nf, v_range[0])
    input = np.genfromtxt(path).reshape(-1)
    X = np.zeros((1, input.size))
    Y = np.zeros((1, 1))
    path = data_path + 'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_min, conf.nf, v_range[0])
    input = np.genfromtxt(path).reshape((dim, dim))
    if conf.input_type == 'ft':
        X[0, :] = fouriertrafo(input).reshape(-1)
    else:
        X[0, :] = input.reshape(-1)
else:
    path = data_path + \
        'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_min, conf.nf, v_range[0])
    input = np.genfromtxt(path).reshape(-1)
    X = np.zeros((1, input.size))
    Y = np.zeros((1, 1))
    path = data_path + \
        'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_min, conf.nf, v_range[0])
    input = np.genfromtxt(path).reshape(-1)
    X[0, :] = input

Y[0, :] = U_min

# fill the arrays of input (X) and output arrays (Y) by looping over all point U along the linescan under consideration
for U_ind in range(len(U_range)):

    U = U_range[U_ind]

    # loop over all samples (in the given case) at a given point U
    for v in v_range:
        if conf.input_type == 'ft' or conf.input_type == 'raw':
            path = data_path + 'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U, conf.nf, v)
            input = np.genfromtxt(path).reshape((dim, dim))

            # when using raw configurations or the magnitude of their discrete FT we additionally need to average over all versions obtained through transformations of p4m
            for n in range(conf.num_trafos):
                X_prov = np.zeros((1, input.size))
                Y_prov = np.zeros((1, 1))

                if conf.input_type == 'ft':
                    X_prov[0, :] = fouriertrafo(symmetrize(input, True)).reshape(-1)
                else:
                    X_prov[0, :] = symmetrize(input, False).reshape(-1)

                Y_prov[0, :] = U

                X = np.append(X, X_prov, axis=0)
                Y = np.append(Y, Y_prov, axis=0)

        else:
            # when considering correlation functions, no symmetry transformations need to be considered
            path = data_path + 'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U, conf.nf, v)
            input = np.genfromtxt(path).reshape(-1)

            X_prov = np.zeros((1, input.size))
            Y_prov = np.zeros((1, 1))

            X_prov[0, :] = input
            Y_prov[0, :] = U

            X = np.append(X, X_prov, axis=0)
            Y = np.append(Y, Y_prov, axis=0)


# perform either standard linear regression or ridge regression on the input (X) and output (Y) arrays
X = np.delete(X, 0, 0)
Y = np.delete(Y, 0, 0)

if conf.regression_type == 'stand':
    reg = LinearRegression().fit(X, Y)
elif conf.regression_type == 'l2':
    reg = Ridge(conf.l2_lambda).fit(X, Y)

Y_pred = reg.predict(X)
Y_pred_U = np.zeros(len(U_range))
Y_true_U = np.zeros(len(U_range))

# obtain the average predictions at each point U by averaging over all corresponding samples
for i in range(len(U_range)):
    step = len(v_range)*conf.num_trafos
    Y_pred_U[i] = np.mean(Y_pred[i*step:i*step+step])
    Y_true_U[i] = np.mean(Y[i*step:i*step+step])

# from the average predictions at each point U along the linescan, calculate the divergence (derivative) of the deviations of the predictions from its true values by a symmetric difference quotient
indicator_array = np.zeros(len(U_range)-2)
for i in range(1, len(U_range)-1):
    delta_l = Y_pred_U[i-1] - Y_true_U[i-1]
    delta_r = Y_pred_U[i+1] - Y_true_U[i+1]
    indicator_array[int(round(i-1))] = (delta_r - delta_l)/(2*U_step)

# plot the derivative and the predictions along the linescan
plot_linescan(U_range, Y_true_U, Y_pred_U, indicator_array, dim)
