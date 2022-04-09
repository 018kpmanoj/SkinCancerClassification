from glob import glob
import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True



# Function to resize image

def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(outpath)

# Resize Training images

input_folder = "/home/karan/Desktop/Melonoma-Classifier/jpeg/train"
output_folder = "/home/karan/Desktop/Melonoma-Classifier/jpeg/resized_train"

images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        (512,512)
    ) for i in tqdm(images)
)


# Resize Testing images

input_folder = "/home/karan/Desktop/Melonoma-Classifier/jpeg/test"
output_folder = "/home/karan/Desktop/Melonoma-Classifier/jpeg/resized_test"

images = glob.glob(os.path.join(input_folder, "*jpg"))
Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        (512,512)
    ) for i in tqdm(images)
)
