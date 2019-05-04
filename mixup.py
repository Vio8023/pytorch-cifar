import numpy as np
import torch

def mixup_data(args, x, y, alpha=1.0, use_uniform=False, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if use_uniform:
        lam = np.random.uniform(0, 1)
    else:
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

    if args.use_post_cutout:
        mask_size_half = args.cutout_size // 2
        offset = 1 if args.cutout_size % 2 == 0 else 0
        for i in range(batch_size):
            if np.random.random() > args.cutout_prob:
                continue
            h, w = 32, 32

            if args.cutout_inside:
                cxmin, cxmax = mask_size_half, w + offset - mask_size_half
                cymin, cymax = mask_size_half, h + offset - mask_size_half
            else:
                cxmin, cxmax = 0, w + offset
                cymin, cymax = 0, h + offset

            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
            xmin = cx - mask_size_half
            ymin = cy - mask_size_half
            xmax = xmin + args.cutout_size
            ymax = ymin + args.cutout_size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            mixed_x[i, :, ymin:ymax, xmin:xmax] = 0
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)