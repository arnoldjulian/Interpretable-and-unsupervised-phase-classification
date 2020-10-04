import numpy as np
from plotting import plot_phase_diagram
from plotting import plot_linescan
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

# range of U under investigation
U_min = conf.U_min
U_max = conf.U_max
U_step = conf.U_step
U_range = np.arange(U_min, U_max+U_step, U_step)

# set path to data folder and specify the range of 'versions', i.e., configurations which should be considered at each point in parameter space
if conf.case == 'noise-free':
    v_range = range(1)
    data_path = conf.path_data_noisefree
elif conf.case == 'noisy':
    v_range = range(1, 11)
    data_path = conf.path_data_noisy

if conf.dim_parameter_space == 1:

    # loop through each point (value of U) along the linescan
    indicator_array = np.zeros(len(U_range))
    U_array = np.zeros(len(U_range)-2)
    for U_ind in range(1, len(U_range)-1):
        U = U_range[U_ind]
        U_l = U_range[U_ind-1]
        U_r = U_range[U_ind+1]

        if conf.case == 'noise-free':
            if conf.input_type == 'corr_func':
                # when considering correlation functions in the noise-free case, no additional averaging needs to be performed to obtain the indicator

                # at each point U, load the input at the points U_r = U+U_step and U_l = U-U_step
                path_l = data_path + \
                    'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_l, conf.nf, v_range[0])

                path_r = data_path + \
                    'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_r, conf.nf, v_range[0])

                input_l = np.genfromtxt(path_l).reshape(-1)
                input_r = np.genfromtxt(path_r).reshape(-1)

                # the indicator at U is the given by the norm of the difference of the inputs at U_r and U_l
                indicator = np.linalg.norm(np.subtract(input_r, input_l))

            else:
                # when using raw configurations or the magnitude of their discrete FT we additionally need to average over all versions obtained through transformations of p4m

                # load input at U_l and U_r
                path_l = data_path + \
                    'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_l, conf.nf, v_range[0])

                path_r = data_path + \
                    'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_r, conf.nf, v_range[0])

                input_l = np.genfromtxt(path_l).reshape((dim, dim))
                input_r = np.genfromtxt(path_r).reshape((dim, dim))

                # calculate the mean input at U_l and U_r by augmenting a given configuration sample through transformations of p4m
                mean_input_l = np.zeros(input_l.shape)
                mean_input_r = np.zeros(input_r.shape)
                if conf.input_type == 'raw':
                    for n in range(conf.num_trafos):
                        mean_input_l = np.add(mean_input_l, symmetrize(input_l, False))

                        mean_input_r = np.add(mean_input_r, symmetrize(input_r, False))
                # when relying on the magnitude of the discrete FT as input, we do not need to consider lattice translations
                else:
                    for n in range(conf.num_trafos):
                        mean_input_l = np.add(
                            mean_input_l, fouriertrafo(symmetrize(input_l, True)))
                        mean_input_r = np.add(
                            mean_input_r, fouriertrafo(symmetrize(input_r, True)))

                mean_input_l = mean_input_l/conf.num_trafos
                mean_input_r = mean_input_r/conf.num_trafos

                # the indicator at U is then obtained from the mean inputs at U_r and U_l
                indicator = np.linalg.norm(np.subtract(mean_input_r, mean_input_l))

        # in the noisy case, we additionally average over all 10 configuration samples available at each point
        elif conf.case == 'noisy':
            if conf.input_type == 'corr_func':

                path_l = data_path + \
                    'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_l, conf.nf, v_range[0])
                input_l = np.genfromtxt(path_l).reshape(-1)

                mean_input_l = np.zeros(input_l.shape)
                mean_input_r = np.zeros(input_l.shape)

                for v in v_range:
                    path_l = data_path + \
                        'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_l, conf.nf, v)
                    path_r = data_path + \
                        'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_r, conf.nf, v)

                    input_l = np.genfromtxt(path_l).reshape(-1)
                    input_r = np.genfromtxt(path_r).reshape(-1)

                    mean_input_r = np.add(mean_input_r, input_r)
                    mean_input_l = np.add(mean_input_l, input_l)

                mean_input_r = mean_input_r/len(v_range)
                mean_input_l = mean_input_l/len(v_range)

                indicator = np.linalg.norm(np.subtract(mean_input_r, mean_input_l))

            else:
                mean_input_l = np.zeros((dim, dim))
                mean_input_r = np.zeros((dim, dim))

                for v in v_range:
                    path_l = data_path + \
                        'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_l, conf.nf, v)

                    path_r = data_path + \
                        'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_r, conf.nf, v)

                    input_l = np.genfromtxt(path_l).reshape((dim, dim))
                    input_r = np.genfromtxt(path_r).reshape((dim, dim))

                    for n in range(conf.num_trafos):
                        if conf.input_type == 'raw':
                            mean_input_l = np.add(mean_input_l, symmetrize(input_l, False))

                            mean_input_r = np.add(mean_input_r, symmetrize(input_r, False))
                        else:
                            mean_input_l = np.add(
                                mean_input_l, fouriertrafo(symmetrize(mean_input_l, True)))
                            mean_input_r = np.add(
                                mean_input_r, fouriertrafo(symmetrize(input_r, True)))

                mean_input_l = mean_input_l/(conf.num_trafos*len(v_range))
                mean_input_r = mean_input_r/(conf.num_trafos*len(v_range))

                indicator = np.linalg.norm(np.subtract(mean_input_r, mean_input_l))

        indicator_array[U_ind] = indicator

    # given the indicator at each point U along the linescan, we subtract the average indicator signal over the entire linescan to account for noise arising due to finite sample statistics
    indicator_array = indicator_array - np.mean(indicator_array)

    # given the indicator at each point U along the linescan, make a plot
    plot_linescan(U_range, indicator_array, dim)

# when considering the full, two-dimensional parameter space, we additionally loop over each value of Nf from 1 to L^2/2 (here, L=20)
elif conf.dim_parameter_space == 2:
    nf_range = range(1, int(round(dim**2/2+1)))

    indicator_array = np.zeros((len(U_range)-2, len(nf_range)))
    p0_array = np.zeros((len(U_range)-2, len(nf_range), 2))
    for nf_ind in range(len(nf_range)):
        for U_ind in range(1, len(U_range)-1):

            nf = nf_range[nf_ind]
            U = U_range[U_ind]
            U_l = U_range[U_ind-1]
            U_r = U_range[U_ind+1]

            if conf.case == 'noise-free':
                if conf.input_type == 'corr_func':

                    path_l = data_path + \
                        'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_l, nf, v_range[0])

                    path_r = data_path + \
                        'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_r, nf, v_range[0])

                    input_l = np.genfromtxt(path_l).reshape(-1)
                    input_r = np.genfromtxt(path_r).reshape(-1)

                    indicator = np.linalg.norm(np.subtract(input_r, input_l))

                else:
                    path_l = data_path + \
                        'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_l, nf, v_range[0])

                    path_r = data_path + \
                        'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_r, nf, v_range[0])

                    input_l = np.genfromtxt(path_l).reshape((dim, dim))
                    input_r = np.genfromtxt(path_r).reshape((dim, dim))

                    mean_input_l = np.zeros(input_l.shape)
                    mean_input_r = np.zeros(input_r.shape)

                    if conf.input_type == 'raw':
                        for n in range(conf.num_trafos):
                            mean_input_l = np.add(mean_input_l, symmetrize(input_l, False))

                            mean_input_r = np.add(mean_input_r, symmetrize(input_r, False))
                    else:
                        for n in range(conf.num_trafos):
                            mean_input_l = np.add(
                                mean_input_l, fouriertrafo(symmetrize(input_l, True)))
                            mean_input_r = np.add(
                                mean_input_r, fouriertrafo(symmetrize(input_r, True)))

                    mean_input_l = mean_input_l/conf.num_trafos
                    mean_input_r = mean_input_r/conf.num_trafos

                    indicator = np.linalg.norm(np.subtract(mean_input_r, mean_input_l))

            elif conf.case == 'noisy':
                if conf.input_type == 'corr_func':

                    path_l = data_path + \
                        'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_l, nf, v_range[0])
                    input_l = np.genfromtxt(path_l).reshape(-1)

                    mean_input_l = np.zeros(input_l.shape)
                    mean_input_r = np.zeros(input_l.shape)

                    for v in v_range:
                        path_l = data_path + \
                            'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_l, nf, v)
                        path_r = data_path + \
                            'U{:.1f}/Nf{:.0f}/v{:.0f}/corr_func.dat'.format(U_r, nf, v)

                        input_l = np.genfromtxt(path_l).reshape(-1)
                        input_r = np.genfromtxt(path_r).reshape(-1)

                        mean_input_r = np.add(mean_input_r, input_r)
                        mean_input_l = np.add(mean_input_l, input_l)

                    mean_input_r = mean_input_r/len(v_range)
                    mean_input_l = mean_input_l/len(v_range)

                    indicator = np.linalg.norm(np.subtract(mean_input_r, mean_input_l))

                else:
                    mean_input_l = np.zeros((dim, dim))
                    mean_input_r = np.zeros((dim, dim))

                    for v in v_range:
                        path_l = data_path + \
                            'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_l, nf, v)

                        path_r = data_path + \
                            'U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U_r, nf, v)

                        input_l = np.genfromtxt(path_l).reshape((dim, dim))
                        input_r = np.genfromtxt(path_r).reshape((dim, dim))

                        for n in range(conf.num_trafos):
                            if conf.input_type == 'raw':
                                mean_input_l = np.add(mean_input_l, symmetrize(input_l, False))

                                mean_input_r = np.add(mean_input_r, symmetrize(input_r, False))
                            else:
                                mean_input_l = np.add(
                                    mean_input_l, fouriertrafo(symmetrize(mean_input_l, True)))
                                mean_input_r = np.add(
                                    mean_input_r, fouriertrafo(symmetrize(input_r, True)))

                    mean_input_l = mean_input_l/(conf.num_trafos*len(v_range))
                    mean_input_r = mean_input_r/(conf.num_trafos*len(v_range))

                    indicator = np.linalg.norm(np.subtract(mean_input_r, mean_input_l))

            indicator_array[U_ind-1, nf_ind] = indicator
            p0_array[U_ind-1, nf_ind, 0] = U
            p0_array[U_ind-1, nf_ind, 1] = nf

    # given the indicator at each sampled point p=(U,Nf) in two-dimensional parameter space, plot the inferred phase diagram
    plot_phase_diagram(p0_array, indicator_array, dim)
