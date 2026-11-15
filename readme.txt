FRIDGE FOOD IDENTIFIER
======================

Goal: take a photo of a fridge and list the food in it.

How it works (use_model.py):
1. Grounding DINO (an open-vocabulary detector) finds objects matching the item names in
   detector_prompts.json and draws a box around each one.
2. SAM 2 outlines each object, which filters out boxes that are too small, too big or empty.
3. Each box is cropped and classified by our MobileNetV2 model into one of 86 food classes.
4. Duplicate filtering keeps one result per item (the detector often boxes the same item
   several times, or boxes a whole group of items).

Current accuracy (Nov 15 2026):
- Classifier on single cropped items: 78.8% on the test split (5,420 images, 86 classes).
- Whole pipeline on 16 real fridge photos (presence F1 vs the team's labels): 29.8%
  (precision 31.6%, recall 28.1%). The first measurement was 16.7%.


SETUP
-----

Python 3.11. Install Git LFS once before cloning or pulling (the dataset and weights use it):

    brew install git-lfs
    git lfs install

Get the SAM 2 code (a git submodule) and its checkpoint (156 MB, not in git):

    git submodule update --init
    curl -L -o checkpoints/sam2.1_hiera_tiny.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

Create a virtual environment with everything for training and running:

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install tensorflow==2.16.2 keras==3.7.0 numpy==1.26.4 opencv-python pillow \
        scikit-learn kaggle kagglehub torch torchvision transformers pillow-heif
    pip install -e sam2_repo

On Apple Silicon Macs also install tensorflow-metal so training runs on the GPU.
Keep numpy at 1.26.x; TensorFlow 2.16 does not work with numpy 2.
Grounding DINO (grounding-dino-tiny, ~660 MB) downloads from Hugging Face on first run.

To run on a machine without the training data (e.g. a Raspberry Pi), skip the dataset:

    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/smartify-fridge/grocery-image-analysis.git
    cd grocery-image-analysis
    git lfs pull --include="trained_model/*"

The pipeline needs about 3 GB of disk and 2-3 GB of RAM. It takes about 5 s per photo
on an M2 Pro GPU and over a minute on a laptop CPU, so expect minutes per photo on a Pi.


MAC GPU BUG (tensorflow-metal)
------------------------------

On a Mac GPU, tensorflow-metal's graph optimizer silently skips the ReLU in
Dense(activation="relu"). Models trained that way give near-zero confidence on any other
machine. MobileNet.fix_metal_relu() turns graph optimization off; main.py and use_model.py
call it automatically.

Weights trained on a Mac GPU before this fix need --no-head-relu in use_model.py.
The weights in trained_model/ were trained after the fix and use the default.


DATASET
-------

combined_dataset/ is already in the repo (via Git LFS), split into train/ val/ test/,
one folder per class. You do NOT need to download anything to train.

It combines 6 sources:
- Grocery Store Dataset (Kaggle: validmodel/grocery-store-dataset)
- Fruits-360 (Kaggle: moltean/fruits), capped at about 160 train images per class
- Ingredient Dataset (Kaggle: keno31/ingredient-dataset)
- Food Ingredient Dataset 51 (Kaggle: sunnyagarwal427444/food-ingredient-dataset-51)
- Open Images V7: real-world crops of fridge foods (Eggs, Cheese, Milk, Bottle, Can,
  Beer, Wine, Bread, plus many fruits and vegetables)
- FoodTypes - individual - in fridge (Roboflow Universe, CC BY 4.0): crops of items in
  real fridges and stores, files named FoodTypes_*. Added by import_detection_crops.py:
  https://universe.roboflow.com/tk-matima-unqyz/foodtypes-individual-in-fridge

FoodTypes cleanup (already applied):
- About a third of its images have speckle noise baked in; those crops are skipped.
- Its generic "Meat" label is mostly chicken and deli packs, so it isn't used for Red Meat.
- 15 sour cream crops labeled yogurt were moved to Sour Cream by hand, and 3 non-yogurt
  crops were deleted. Re-importing FoodTypes would undo this.

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
- Saves weights to combined_mobilenet.weights.h5 and rewrites class_names.json
  (in the top folder, not trained_model/).
- Takes about 2 hours on an M2 Pro. caffeinate -i keeps the Mac awake, but closing the
  lid still puts it to sleep.

To use the new model, copy it into trained_model/ (what use_model.py loads):

    cp combined_mobilenet.weights.h5 class_names.json trained_model/

Options:
- -m lenet     train the small LeNet model instead (64x64, weights in combined.weights.h5)
- -l 1         start from the saved weights (only works if the class list hasn't changed)
- -s 1         LeNet only: ignore fruits360_pretrain.keras and train from scratch


RUNNING ON PHOTOS
-----------------

Phone photos are often HEIC or rotated. Convert them first:

    python convert_img_format.py --input test_images --output test_images_converted

Then:

    python use_model.py --images test_images_converted --output model_output

Output in model_output/:
- detections.csv   every detection, with status kept / duplicate / rejected
- summary.csv      the items found in each photo
- annotated/       each photo with the kept items drawn
- crops/           the crops of kept items (--save-all-crops saves every crop)

Main options:
- --classifier-threshold   minimum classifier confidence (default 0.30)
- --box-threshold          minimum Grounding DINO confidence (default 0.30)
- --box-nms, --mask-nms    overlap at which two items count as the same (0.50, 0.60)
- --keep-duplicates        skip duplicate filtering (raw detector output)
- --crop-mode mask         classify SAM 2 masked crops instead of boxes (scored worse)
- --weights, --classifier-classes   use a different model
- --no-head-relu           for weights trained on a Mac GPU before the ReLU fix

The detector's item names are in detector_prompts.json. Adding a name there lets the
detector find it without retraining, but the classifier can only answer with its 86 classes.


MEASURING ACCURACY ON FRIDGE PHOTOS
-----------------------------------

    python evaluate_fridge.py --detections model_output/detections.csv --presence

fridge_ground_truth.csv has the team's labels for the 16 photos in test_images/
(one row per class per photo). Items with no matching class are listed but not scored.
--presence scores whether each class is in each photo; without it, counts must match too.


KNOWN LIMITATIONS
-----------------

- Only 16 labeled fridge photos, so the 29.8% is a rough number.
- Yogurt tubs are mostly misclassified (as Milk, Sour Cream or Can). About half the Yogurt
  training images are drinkable yogurt cartons; more photos of tubs would help most.
- The detector almost never boxes chicken, grapes, juice, berries or onions with those
  names, so the classifier never sees them. More specific prompts may help.
- Packaged items are often called Bottle or Can. Jars (pickles, salsa) have no class.
- No training data yet for butter, condiments (ketchup, mustard, mayo) or soda.
- Some FoodTypes products appear in both train and test, so 78.8% may be slightly high.


FILES
-----

main.py                    train the classifier
use_model.py               detect and classify food in photos
evaluate_fridge.py         score use_model.py output against fridge_ground_truth.csv
mobilenet.py, lenet.py     model definitions (mobilenet.py has the Mac GPU fix)
data_combined.py           loads combined_dataset/ for training
combine_datasets.py        builds combined_dataset/ and holds the class mapping
download_open_images.py    downloads Open Images crops into open_images/
import_detection_crops.py  crops a Roboflow COCO export (FoodTypes) into combined_dataset/
convert_img_format.py      converts phone photos (HEIC, rotation) for use_model.py
compare_detection_model.py compares YOLO and Grounding DINO on test photos
data_ingredients.py        downloads the two Kaggle ingredient datasets
data_grocery.py, data_v2.py  older loaders for the Grocery Store and Fruits-360 datasets
detector_prompts.json      item names given to Grounding DINO
fridge_ground_truth.csv    team labels for the fridge test photos
trained_model/             the weights and class list use_model.py loads
class_names.json           class order written by the last training run
