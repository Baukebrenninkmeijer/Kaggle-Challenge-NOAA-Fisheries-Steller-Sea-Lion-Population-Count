'''
Het idee is om uit de data patches te halen op het moment dat deze nodig zijn
voor het trainen. Ook wordt er 'data augmentation' toegepast zodat het neurale
netwerk niet overfit op de trainin data en beter generaliseert. 
'''

from PIL import Image
import random
import numpy as np
import pandas as pd 
import os
from scipy.misc import imresize
from glob import glob
from scipy.ndimage.interpolation import rotate

from matplotlib import pyplot as plt

class PatchGenerator:

    '''
      files :: dictionary
          files[file_number] = fullpath to file

      labels :: list
          labels[0] = (file_number, class, (x, y))
    '''
    def __init__(self, files, labels, batch_size, num_classes=2, patch_size=140, random_flipping=False, max_shift=0, max_rotation = 0, floatX='float32',downsample=False):
        self.files = files
        self.labels = labels
        self.num_classes = num_classes
        self.random_flipping = random_flipping
        self.max_shift = max_shift
        self.max_rotation = max_rotation
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.floatX = floatX
        self.downsample = downsample
        
        # max_shift should be a positive int or zero
        assert (max_shift >= 0), "Max shift should be a positive integer or zero."
        
        # Batch size should be even
        assert (batch_size % 2 == 0),"Batch size should be even in order to be able to generate balanced batches."
        
        # Keeps track of the index in the labels array where the next batch
        # will start.
        self.batch_index = 0
        
        # Store the most recently loaded images such that we don't do
        # any redundant loading of images
        self.cache = {}
        
    def __iter__(self):
        return self

    def next_labels_pos(self):
        """
            Return the positive labels for the current batch.
            The next batch will start at the end of the current batch.
            If the current batch reaches the end of the training set,
            start at the beginning.
        """
        batch_end = self.batch_index + self.batch_size // 2
        if (batch_end >= len(self.labels)):
            self.batch_index = 0
            batch_end = self.batch_size // 2

        batch_labels = self.labels[self.batch_index:batch_end]
        self.batch_index = batch_end
        return batch_labels
    
    def next_labels_neg(self):
        """
            Return the negative labels for the current batch.
            The negative samples are drawn randomly from the first image
            currently loaded in cache, with sufficient distance from a 
            positive ground truth.
        """
        img_id = self.cache.keys()[0]
        img = self.cache[img_id]
        ground_truth_pos = np.zeros(img.shape,dtype=np.uint8)
        ground_truth_neg = np.ones(img.shape, dtype=np.uint8)

        sizeout = 500
        sizein = 50
        
        if self.downsample:
            pos_coords = [(l[2][0]//2,l[2][1]//2)for l in self.labels if l[0] == img_id]
        else:
            pos_coords = [l[2] for l in self.labels if l[0] == img_id]
        
        # Create a 80x80 square around the positive ground truth
        # from which we won't create negative samples
        # todo: Create a NxN square around positive ground truth
        for (x,y) in pos_coords:
            ground_truth_neg[max(0, x - sizeout):min(x + sizeout, img.shape[0]),
                             max(0,y-sizeout):min(y+sizeout,img.shape[1])] = 0

        for (x, y) in pos_coords:
            ground_truth_neg[max(0, x - sizein):min(x + sizein, img.shape[0]),
            max(0, y - sizein):min(y + sizein, img.shape[1])] = 1

        neg_idxs = np.where(ground_truth_neg==0)
        # neg_idxs = np.where(ground_truth_pos==0)
        rand_idxs = np.random.randint(0,len(neg_idxs[0]),self.batch_size//2)
        # plt.imshow(img)
        # plt.imshow(ground_truth_neg*120,alpha=0.9)
        # plt.show()
        return [(img_id,-1,(neg_idxs[0][i],neg_idxs[1][i])) for i in rand_idxs]
    
    def clean_cache(self):
        """
            Delete the images in the cache that are no longer needed.
            Call this function BEFORE calling next_labels, otherwise
            the images for the labels that are returned in that function 
            are removed from cache.
        """
        current_img_id = self.labels[self.batch_index][0]
        for key in self.cache.keys():
            if (key < current_img_id):
                self.cache.pop(key)
    
    def load_images(self, image_ids):
        """
            Load images with given ids if they are not cached.
            Store the loaded images in the cache.
        """
        for img_id in image_ids:
            if not (img_id in self.cache):
                # Load the image using PIL and store it in the cache
                print("Loading image {}".format(img_id))
                img_path = self.files[img_id]
                image = np.array(Image.open(img_path))
                if (self.downsample):
                    image = imresize(image,0.5)
                self.cache[img_id] = image/255.0
    
    def normalize_image(self, image):
        """
            Normalize an image per-channel.
        """
        means = np.mean(image, axis=(0,1))
        stds = np.std(image, axis=(0,1))
        return (image-means)/stds
        
    
    def to_one_hot(self, label):
        """
            Convert the given label to one-hot.
            -1 as input is considered negative.
            If self.num_classes = 2, all positive labels will be grouped.
        """
        label_oh = np.zeros((self.num_classes),dtype=np.uint8)
        if label == -1:
            label_oh[0] = 1
        elif self.num_classes == 2:
            label_oh[1] = 1
        else:
            label_oh[label+1] = 1
        return label_oh
    
    def next(self):
        '''Return the next batch of shape (batch_size, patch_size, patch_size, 3)'''
        # First, clean the cache
        self.clean_cache()
        # Get the labels for the current batch
        batch_labels_pos = self.next_labels_pos()
        
        # Get the file numbers that we need to load for this batch only
        file_nrs = list(set([l[0] for l in batch_labels_pos]))
        
        # Load the images into the cache
        self.load_images(file_nrs)
        
        # Get negative samples
        batch_labels_neg = self.next_labels_neg()
        
        batch_size = self.batch_size
        patch_size = self.patch_size
        
        X = np.zeros((batch_size,3, patch_size, patch_size),dtype=self.floatX)
        Y = np.zeros((batch_size, self.num_classes), dtype=np.uint8)
        
        for i, (img_id, label, pos) in enumerate(batch_labels_pos):
            # The image should be loaded in the cache
            img = self.cache[img_id]
            
            # extract the patch with the label in the center
            offset = patch_size // 2

            x, y = pos
            
            if self.downsample:
                x = x//2
                y = y//2

            dx = 0
            dy = 0
    
            m = self.max_shift
            if m > 0:
                # random shifts of m pixels maximum
                dx = random.randint(0, 2 * m) - m
                dy = random.randint(0, 2 * m) - m
    
            x += dx
            y += dy

            # slice img om een patch te krijgen
            # TODO: pad met zeros als edge case?
            # Probleem: de zeeleuw moet nog wel in het midden zitten
            # Probleem: information leakage als we alleen
            # de positieve class padden..
            xdiff = -(x-offset) if (x-offset < 0) else 0
            ydiff = -(y-offset) if (y-offset < 0) else 0
            patch = img[max(0,x-offset):min(x+offset,img.shape[0]), 
                        max(0,y-offset):min(y+offset,img.shape[1]), :]
            
            # if patch.shape != (patch_size, patch_size, 3):
            #     print('Edge case, fix this...')
    
            # random flip?
            if self.random_flipping:
                if random.choice([True, False]):
                    patch = np.flipud(patch)
                if random.choice([True, False]):
                    patch = np.fliplr(patch)
                    
            # random rotation? 
            theta = 0
            if self.max_rotation > 0:
            #create a random rotation matrix 'rot_mat' in the range of -'self.max_rotations' to +'self.max_rotations'
                theta = random.uniform(-self.max_rotation, self.max_rotation)

            patch[:,:,0] = rotate(patch[:,:,0], theta, reshape = False)
            patch[:,:,1] = rotate(patch[:,:,1], theta, reshape = False)
            patch[:,:,2] = rotate(patch[:,:,2], theta, reshape = False)
            
            
            
            # Hier gebeurt impliciet de zero padding
            # De index is hier * 2 zodat alle even indices positieve samples zijn
            X[i*2, :,xdiff:xdiff+patch.shape[0],ydiff:ydiff+patch.shape[1]] = patch.transpose((2,0,1))
            Y[i*2,:] = self.to_one_hot(label)
            
        # Get negative patches
        for i, (img_id, label, pos) in enumerate(batch_labels_neg):
            # The image should be loaded in the cache
            img = self.cache[img_id]
            
            # extract the patch with the label in the center
            offset = patch_size // 2

            x, y = pos
            
            xdiff = -(x-offset) if (x-offset < 0) else 0
            ydiff = -(y-offset) if (y-offset < 0) else 0
            patch = img[max(0,x-offset):min(x+offset,img.shape[0]), 
                        max(0,y-offset):min(y+offset,img.shape[1]), :]
            
            #if patch.shape != (patch_size, patch_size, 3):
            #    print('Edge case, fix this...')
                
            # Hier gebeurt impliciet de zero padding
            # De index is hier * 2+1 zodat alle oneven indices 
            # negatieve samples zijn
            X[i*2+1,:,xdiff:xdiff+patch.shape[0],ydiff:ydiff+patch.shape[1]] = patch.transpose((2,0,1))
            Y[i*2+1] = self.to_one_hot(label)

        return X, Y


classes = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']

if __name__ == '__main__':
    train_files_path = os.path.join('TrainSmall2', 'Train', '*.jpg')
    train_files = sorted(glob(train_files_path))

    coordinates_path = 'correct_coordindates.csv'
    csv = pd.read_csv(coordinates_path,index_col=0)
    
    # op deze manier gebruiken we alleen labels voor train_files op... 
    files = dict()
    # List of tuples (img, class, (row,col))
    labels = []

    for fullpath in train_files:
        prefix, fname = os.path.split(fullpath)
        name, ext = os.path.splitext(fname)

        file_nr = int(name)

        # 
        files[file_nr] = fullpath

        labels_nr = csv[csv['filename'] == fname]

        for _, row in labels_nr.iterrows():
            label_class = classes.index(row['category'])
            (x, y) = int(row['y_coord']), int(row['x_coord']) # logical?
            labels.append( (file_nr, label_class, (x, y)) )
    
    # files, labels should be okay
    batch_size = 24
    
    # Wietse: zet hier num_classes=len(classes)+1
    pgen = PatchGenerator(files, labels, batch_size)

    for j in range(2):
        plt.figure(figsize=(8,10))
        plt.title('Randomly generated patches')
    
        X,Y = next(pgen)
        for i in range(batch_size):
            plt.subplot(6,4,i+1)
            plt.title('Class {}'.format(np.argmax(Y[i])))
            plt.imshow(X[i].transpose((1,2,0)))
        
        plt.tight_layout()
        plt.show()