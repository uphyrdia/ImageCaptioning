"""Download the Flickr8k dataset (8,091 images + captions.txt) from Kaggle."""
import kagglehub

path = kagglehub.dataset_download("adityajn105/flickr8k")
print(path)
