# Train CIFAR10 with PyTorch

To run the best model (combining mixup, cutout, shake-shake) mentioned in our report, simply run

python main.py --wd=1e-4 --train --use_post_cutout --use_mix_│mixup_uniform_9185_ResNet20_start_epoch_0_epochs_160.pk
up --model_arch=SSResNet20 --nepochs=320

Some other available parameters can be seen in the code.

Default is to train ResNet-20, to run other models, please modify the model name in the code.

The initial codebase structure referred to the implementation of https://github.com/kuangliu/pytorch-cifar.
