README

################################################

Original Dataset:
	
	13868 Images are found out of the 15k original images. Use beauty_new.csv for the 13868 images. 

	Columns: 

	'#flickr_photo_id': Photo Id on Flickr. Used to query the image.
	'category': Category of the photo. Not used.
	'beauty_scores' : a list of the scores this image received.


################################################

download_image.py

Queries the images using the "beauty.csv" file and then saves the images in the '\jpg' directory. Skips if the image is missing or can't be queried.

################################################

Link to the original Image Aesthetic Dataset: 

http://www.di.unito.it/~schifane/dataset/beauty-icwsm15/

################################################

Preprocessing

	###############

	calc_beauty.py

	Replace the 'beauty_scores', which is a column where each image has a list of scores, with the mean score of the image. The means scores are then rounded to the nearest integer within range [1,5]. To maintain consistency among datasets, we then shift all the scores to 1 score lower, so that we have a score range of [0,5]. 

	Output is saved in 'aes.csv'.

	##############
	
	split_data.py

	Randomly split the 'aes.csv' file into train(0.8), valid(0.05), test datasets(0.15).

	##############