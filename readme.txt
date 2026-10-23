FRIDGE FOOD IDENTIFIER
======================

Goal: take a photo of a fridge and list the food in it.

How it works (use_model.py):
1. A YOLO detector finds objects in the photo and draws a box around each one.
2. Each box is cropped and classified by a MobileNetV2 model into one of 86 food classes.

Current classifier accuracy: 77.2% on the test split (5,089 cropped images, 86 classes).
This is accuracy on single cropped items, not on whole fridge photos.


SETUP
-----

Python 3.11. Use the mlenv virtual environment (it has everything needed for training):

    source mlenv/bin/activate

If you need to recreate it:

    python3.11 -m venv mlenv
    source mlenv/bin/activate
    pip install tensorflow keras numpy opencv-python pillow scikit-learn kaggle kagglehub

On Apple Silicon Macs also install tensorflow-metal so training runs on the GPU.

use_model.py also needs ultralytics (not installed in mlenv yet):

    pip install ultralytics

The dataset and weights are stored with Git LFS. Install it once before cloning or pulling:

    brew install git-lfs
    git lfs install


DATASET
-------

combined_dataset/ is already in the repo (via Git LFS), split into train/ val/ test/,
one folder per class. You do NOT need to download anything to train.

It combines 5 sources:
- Grocery Store Dataset (Kaggle: validmodel/grocery-store-dataset)
- Fruits-360 (Kaggle: moltean/fruits), capped at about 160 train images per class
- Ingredient Dataset (Kaggle: keno31/ingredient-dataset)
- Food Ingredient Dataset 51 (Kaggle: sunnyagarwal427444/food-ingredient-dataset-51)
- Open Images V7: real-world crops of fridge foods (Eggs, Cheese, Milk, Bottle, Can,
  Beer, Wine, Bread, plus many fruits and vegetables)

The original 193 classes were merged into 86 broader fridge classes (e.g. all apple
varieties -> Apple, all milk brands -> Milk). Pantry items (spices, nuts, honey) were
dropped. The mapping is BROAD_CLASSES in combine_datasets.py.
class_names.json lists the 86 classes in the order the model uses.


TRAINING
--------

    caffeinate -i python main.py -m mobilenet

- Trains MobileNetV2 (pretrained on ImageNet) at 192x192 in two phases:
  1. train only the new classifier layers (up to 15 epochs)
  2. fine-tune the top of MobileNetV2 (up to 40 epochs)
- Uses class weights (rare classes count more) and label smoothing (0.1).
- Saves weights to combined_mobilenet.weights.h5 and rewrites class_names.json.
- Takes about 1.5-2 hours on an M2 Pro. caffeinate -i keeps the Mac awake.

Options:
- -m lenet     train the small LeNet model instead (64x64, weights in combined.weights.h5)
- -l 1         start from the saved weights (only works if the class list hasn't changed)
- -s 1         LeNet only: ignore fruits360_pretrain.keras and train from scratch


RUNNING ON PHOTOS
-----------------

    python use_model.py -i path/to/photo.jpg
    python use_model.py -i path/to/folder_of_photos --save-visualization

Options:
- --save-visualization      save a copy of each photo with boxes drawn (*_detected.jpg)
- --detector-confidence     minimum YOLO confidence (default 0.25)
- --classifier-confidence   minimum classifier confidence (default 0.30)
- --weights                 use a different weights file

yolo11n.pt is downloaded automatically by ultralytics on first run.


REBUILDING THE DATASET (only if you change classes or sources)
-------------------------------------------------------------

Most changes only need the existing dataset plus the Open Images crops:

    python download_open_images.py
    python combine_datasets.py --from-combined --overwrite

- download_open_images.py saves about 16k crops (~500 MB) to open_images/. It streams
  Open Images' annotation files instead of saving them. Use --per-class to change how
  many images per class, and --splits validation for a quick test run.
- --from-combined rebuilds from the current combined_dataset/ (keeping its train/val/test
  splits) plus open_images/, then applies the class mapping and drops pantry items.

Rebuilding from the raw Kaggle downloads (python combine_datasets.py --overwrite) does
NOT currently reproduce combined_dataset/. The current Fruits-360 download uses different
class names and has far more images, and some raw folder names aren't in the mapping.
Use --from-combined instead until that is fixed.

After any rebuild, retrain the model. Old weights won't load if the class list changed.


KNOWN LIMITATIONS
-----------------

- The detector (yolo11n.pt) was trained on COCO, which only knows about 10 foods. It often
  misses milk cartons, eggs, cheese, yogurt and containers, so those never get classified.
  Planned fix: switch to YOLO-World prompted with our class names.
- No training data yet for butter, condiments (ketchup, mustard, mayo) or soda.
- Weakest classes: Ginger, Lettuce, Lime, Sausage, Wine, Sour Cream.
- Most common mix-ups: Mandarin/Orange, Grapefruit/Orange, Zucchini/Cucumber, Wine/Beer.
- The 77% score is on cropped items from the dataset. Accuracy on real fridge photos still
  needs to be measured with a fridge test set.


FILES
-----

main.py                  train the classifier
use_model.py             detect and classify food in photos
mobilenet.py, lenet.py   model definitions
data_combined.py         loads combined_dataset/ for training
combine_datasets.py      builds combined_dataset/ and holds the class mapping
download_open_images.py  downloads Open Images crops into open_images/
data_ingredients.py      downloads the two Kaggle ingredient datasets
data_grocery.py          older loaders for the Grocery Store and Fruits-360 datasets
class_names.json         class order used by the saved weights
