# JointLibras
Here are the models tested to find body joint coords in images.

## Dataset processing
Some of them proccess the dataset in them (the ones that uses the youtube
dataset): Model 002

Others just use the data after using a dataset proccess that can be found in
`/dataset_process` folder: Models 001, 003, 004 and 005.

### Dataset load checking
The dataset are all given in matlab format, that is mannualy "translated" to python
and may have errors, the data checkers render some images with the joints to see
if they are correct, until now there is only to BBC dataset, but the result checker
can be used to achieve the same test with the others datasets.  
\* All the results are put on `/load_data_check` folder.

### Result checking
The same as the dataset cheking, but at the final of the training, a result checker
is called to render some images with the joints prediction from the models.  
\* All the results are put on `/result_check` folder.

## Folder structure
Those that uses a preprocessed data get the image paths from a `image.txt` file
on root (yeah, root of the disk, if needed you can use a symlink, pass `./` doesn't work) and for the joints values to train/validation it uses the `joint.npy`
(binary file numpy array).

This thing about root do a little mess, but it's caused by `image_preloader` from
[tflearn](http://tflearn.org), and it's needed to free a lot of RAM memory (It
took almost 10Gb on a folder with 17k 128x128 images without the preloader).

## BBC Pose Dataset
The most complete dataset i found was BBC Pose Dataset, that can be found here:
[BBC Dataset](https://www.robots.ox.ac.uk/~vgg/data/pose/index.html#downloadlink)
It's using only the test parte of folder 1 from dataset, but it can be changed setting
the `test` variable in `dataset_process/bbc.py` to `False`.
The eval matlab file that comes in the dataset should be renamed to `dataset.mat`
and pasted on `dataset_bbc` folder and the images should be pasted in `dataset_bbc/data/1` folder.


# Models
The first two models were based on [Heterogeneous Multi-task Learning for Human Pose Estimation with Deep Convolutional Neural Network](https://arxiv.org/abs/1406.3474) (without the window and human body detector parts), but they were tweaked a lot to work.
The others 003 and 004 are based on the first ones and with a lot of empirical
tryings.
And the last one (005) was used an AlexNet architecture.

# Results
All of them is getting the same result... It predicts the same joint spots to
any the input data passed to them.
