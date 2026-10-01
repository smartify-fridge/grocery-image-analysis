Python version 3.11, some libraries might not be compatible with higher versions of Python as of Dec. 13 2024

Libraries downloaded in virtual enviorment using pip install:
- keras
- tensorflow
- kaggle
- numpy
- opencv

Dataset: Grocery Store Dataset (https://www.kaggle.com/datasets/validmodel/grocery-store-dataset).
For this code to run, an API key from Kaggle (~/.kaggle/kaggle.json) is needed to download the dataset.
On first run it is downloaded automatically into ./grocery_kaggle.

To run:
- run Python main.py in bash for the first time
- run Python main.py -l 1 to load previous saved weights
- add -c 1 to train on the 42 coarse classes (e.g. Apple) instead of the 81 fine classes (e.g. Granny-Smith)
- run Python main.py -p 1 to first pretrain on Fruits-360 (https://www.kaggle.com/datasets/moltean/fruits)
  (downloads an 8GB zip, keeps only the 100x100 images in ./fruits360_kaggle, then deletes the zip)
  the pretrained model is saved to fruits360_pretrain.keras and later runs start from it automatically
- add -s 1 to ignore the pretrained model and train from scratch
- run Python main.py -m mobilenet to use MobileNetV2 pretrained on ImageNet instead of LeNet
  (downloads ~9MB of ImageNet weights on first run, uses 160x160 images, weights saved to grocery_mobilenet_fine.weights.h5)
  -c 1 and -l 1 work with it too; -p and -s only apply to LeNet
