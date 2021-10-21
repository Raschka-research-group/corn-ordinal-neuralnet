import flickrapi
import pandas as pd
import urllib
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

api_key = "91dc9b1c17106ef8184b38acc4dfba7f"
api_secret = "001f5f03fee6a4e8"
# Flickr api access key 
flickr = flickrapi.FlickrAPI(api_key, api_secret)
df = pd.read_csv('./beauty.tsv', sep='\t')
def get_url(pid):
    #generate flickr URL from given photo ID.
    try:
        root = flickr.photos_getInfo(photo_id=pid)
    except flickrapi.exceptions.FlickrError:
        return ""
    info = root[0].attrib
    pid = info["id"]
    server = info["server"]
    secret = info["secret"]
    url = "https://live.staticflickr.com/" + server + "/" + pid + "_" + secret + ".jpg"
    return url

added = False

def fetch_save(pid,added):
    #Citing the fetch code from the fetch-dataset.ipynb
    #Returns True if downloaded the photo sucessfully
    added = False
    url = get_url(pid)
    if url == "":
    	return False
    #format of saved photos: <flickr_photo_id>.jpg
    filename_url = str(pid)+".jpg"
    file_dest_index = os.path.join('jpg', filename_url)
    if not os.path.exists(file_dest_index):
        
        try:
            request = urllib.request.urlopen(url, timeout=15)
        except urllib.error.HTTPError:
            return False
        with open(file_dest_index, 'wb') as f:
            try:
                f.write(request.read())
            except:
                print(f"error in {filename}")
                return False
        im = Image.open(file_dest_index)
        im.save(file_dest_index)
        added = True
    return added
newdf = pd.DataFrame(columns=df.columns)
for i in range(df.shape[0]):
    pid = df.iloc[i][0]
    added = fetch_save(pid,added)
    if added:
        newdf = newdf.append(df.iloc[i],ignore_index = True)
    if i%100 == 0:
        print(i)
newdf.to_csv("beauty_new.csv")


# fetch_save(df.iloc[2][0])
