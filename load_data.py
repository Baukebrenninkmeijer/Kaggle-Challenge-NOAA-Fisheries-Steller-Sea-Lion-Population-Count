from PIL import Image
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os


images = []
files = []
images_dotted = []
files_dotted = []

train_files_path = os.path.join('TrainSmall2', 'Train', '*.jpg')
dotted_files_path = os.path.join('TrainSmall2', 'TrainDotted', '*.jpg')
labels_path = os.path.join('TrainSmall2', 'Train', 'train.csv')

for filename in glob.glob(train_files_path):
    print(filename)
    im = Image.open(filename)
    images.append(im)
    files.append(filename)

for filename in glob.glob(dotted_files_path):
    print(filename)
    im = Image.open(filename)
    images_dotted.append(im)
    files_dotted.append(filename)

labels = pd.read_csv(labels_path, index_col=0)

plt.imshow(images[0])
plt.show()

