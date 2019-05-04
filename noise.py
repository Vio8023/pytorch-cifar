import numpy as np
import torch

def noise_data(noise_type, noise_prob):
    """
    noise_type: the noise function, can be gauss or s&p
    noise_prob: the probability of the noise function, by default 1.0 implies no noise applied.
    """
    if noise_type == "gauss":
        def _noise(image):
            s = np.random.uniform()
            if s < noise_prob:
                return image
            image = image.numpy()
            mean, sigma = 0, 0.1
            gauss = np.random.normal(mean, sigma, image.shape)
            noisy = image + gauss
            return torch.FloatTensor(noisy)

    elif noise_type == "s&p":
        def _noise(image):
            s = np.random.uniform()
            if s < noise_prob:
                return image
            image = image.numpy()
            s_vs_p = 0.5
            amount = 0.004
            out = image.copy()
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[tuple(coords)] = 1
            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[tuple(coords)] = -1
            return torch.FloatTensor(out)

    else:
        raise ValueError("the noise type {} is not defined".format(noise_type))

    return _noise

def recover_image(input_array):
    input_array = input_array.numpy()
    data = input_array * np.expand_dims(np.expand_dims(np.array((0.247, 0.243, 0.261)), 1), 2)
    data += np.expand_dims(np.expand_dims(np.array((0.4914, 0.4822, 0.4465)), 1), 2)
    data = torch.FloatTensor(data)
    return data

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
