#!pip install tqdm

import requests
import math
import os
import zipfile
from tqdm import tqdm_notebook as tqdm

if not os.path.exists(os.path.join(r"..\\data\\", "PetImages")):
    url = "https://downloads.codingcoursestv.eu/037%20-%20neuronale%20netze/PetImages.zip"
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0));
    block_size = 1024

    print("Downloading...")
    with open(os.path.join(r"..\\data\\", "PetImages.zip"), 'wb') as f:
        for data in r.iter_content(block_size):  #, total=math.ceil(total_size // block_size), unit='KB',
                                                # unit_divisor=1024, unit_scale=True):
            f.write(data)

    print("Download completed")
    print("Extracting...")

    zip_ref = zipfile.ZipFile(os.path.join(r"..\\data\\", "PetImages.zip"), 'r')
    zip_ref.extractall(os.path.join(r"..\\data\\"))
    zip_ref.close()

    print("Done!")
else:
    print("Datei existiert bereits")