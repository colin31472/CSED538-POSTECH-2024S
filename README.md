# Deeplearning_prj
Repository for deeplearning class project

How to run code? (in python 3.):: 

pip install -r requirements.txt

python main.py --config='./configs/example.py'

Learning rate scheduler specify::


In configs/example.py, you can specify our learning rate scheduler with lr_scheduler = loss_informed_scheduler.
In config.new_technique_args, you can specify factors, batch sizes, and initial learning rate. Please refer to example code for more details.

How to reproduce our results? ::
- For CIFAR-10 with factor 5: python main.py --config='./configs/factor_5.py'
- For CIFAR-10 with batch 32: python main.py --config='./configs/batch32.py'

Using other config files to reproduce for other number of factors or batch sizes, and use lr.py as config file to check the result of initial learning rate = 1


- For CIFAR-100, you have to change little bit of our code. In data_loader.py, you have to change dataset download code to datasets.CIFAR100, and you have to change num_classes=10 to num_classes100 in models/ResNet.py. After then you can run the code with this command line:
- For already existing pre-determined schemes, you have to specify in config file (see CosineAnnealing for example) as config.lr_scheudler = {Pytorch learning rate scheduler object}. Please refer to our example configuration file.






Reference codes

Pytorch's learning rate scheuler codes are refered to implement loss_informed_scheduler.

Please refer to https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR for original code.

Also, we refered https://github.com/NvsYashwanth/CIFAR-10-Image-Classification/blob/master/cifar10.ipynb and https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py to implement trainer, dataloader, and evaluate code.




