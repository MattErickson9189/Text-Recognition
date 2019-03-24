This is my research project for my senior computer science capstone project. It uses supervised learning with a neural net to identify hand writing.

Before running the models you first must run the two data processing scripts

1. First run ImageProcesor.py. This goes through all the images in the IAM database
resizes, then saves them with a new name to the Resized/ directory

2. Then in order to filter out all the images with incorrect labels, you need to run the DataCleaner.py script to remove all of the damaged images

