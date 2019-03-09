# Train CIFAR10 with PyTorch

To run the code, simply run

python main.py --lr=0.1 --wd=1e-4

Some other available parameters can be seen in the code.

Default is to train ResNet-20, to run other models, please modify the model name in the code.

The code for model architecture (resnet and resnext) are modified from  https://github.com/kuangliu/pytorch-cifar.
