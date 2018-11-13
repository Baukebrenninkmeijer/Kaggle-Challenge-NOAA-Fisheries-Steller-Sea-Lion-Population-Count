# Sea Lion Population Count

This project was originally hosted on the Radboud University Gitlab. For archival purposes I copied it here. 

## Project proposal - Disqualified
Contributers
- Bauke Brenninkmeijer
- Timo van Niedek
- Démian Janssen
- Emma Gerritsen
- Ties Robroek
- Wietse Kuipers

Submission for the Kaggle Challenge: [NOAA Fisheries Steller Sea Lion Population Count](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count).

### Data

- Small training set (89 MB)
	- [The TrainSmall.7z archive does not seem to follow the pattern of train and dotted images correctly. For example, see Train/3.jpg and TrainDotted/3.jpg. The same thing for 7, 9, etc.](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/30746)
    - Download with `scp sciencelogin@lilo4.science.ru.nl:/scratch/zeeleeuw.zip . `
- Huge training set (95 GB)
- Labels:
	- Counts per type:
	- Adult males
	- Subadult males
	- Adult females
	- Juveniles
	- Pups
- Train-dotted is included with colored dots placed on top of each sea lion
	- Color is based on type
	- [Someone created a script on Kaggle (at the Kernels tab) that extracts coordinates for sea lions](https://www.kaggle.com/threeplusone/sea-lion-coordinates)
- [There are way more females than males in the training set](https://www.kaggle.com/philschmidt/sea-lion-correlations-cv2-template-matching)

### Methods from other participants

- [A model with 0.91 accuracy](https://www.kaggle.com/radustoicescu/use-keras-to-classify-sea-lions-0-91-accuracy) uses simple blob detection (skimage.feature.blob_log). The blobs are classified using a CNN with 2 convlayers and one fully connected layer.
	- The comments say that the model is overfitting.
- [More blob detection](https://www.kaggle.com/radustoicescu/get-coordinates-using-blob-detection)

### Proposed method

- Train first on the small training subset for agile development iterations; then use the large set on surfsara
- We might need to do a segmentation, extract patches and classify the patches
- Prior distribution of sea lions found [here](https://www.kaggle.com/philschmidt/sea-lion-correlations-cv2-template-matching) might be useful. We don't know for sure if the test set follows the same distribution though.
- Simple blob detection using skimage.feature.blob_log first; then if necessary machine learned segmentation.
- CNN for classifying patches
- Investigate attention-based systems

### Tasks

- Data loading	(team 1)
- Data augmentation (team 1)
	- Rotate, shift, elastic deformation, blur, noise
- Coordinate matching [(use this)](https://www.kaggle.com/threeplusone/sea-lion-coordinates) (team 2)
- Segmentation or blob detection (team 2)
	- Check skimage.feature.blob_log first to see if it corresponds with the TrainDotted files; otherwise we may have to use something more sophisticated
- Patch classification (for later)
	- CNN (team 1)
	- Attention-based systems (team 2)

Team 1: Wietse, Ties, Emma

Team 2: Bauke, Demian, Timo


### Update tasks:

- Démian: patch extractor
- Timo, Emma & Ties: FCN segmentation
- Wietse: Paper lezen Maaike
- Bauke: Plan maken zeeleeuwen classificatie


### Options for classification
 - Bag of Words
 - sparse coding spatial pyramid matching (ScSPM)
 - Gewoon de segmented images in een neural network gooien

### Inloggegevens voor Surfsara
Username: gdemo055
Wachtwoord: Ml1pisawesome!
