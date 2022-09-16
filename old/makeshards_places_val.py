import os
import os.path
import random
import argparse
import webdataset as wds
from multiroot_image_folder import MultirootImageFolder

parser = argparse.ArgumentParser('Generate sharded dataset from a list of image directories')
parser.add_argument('--filekey', action='store_true', help='use file as key (default: index)')
parser.add_argument('--maxsize', type=float, default=1e16, help='max size of each shard (set big if you want a single shard)')
parser.add_argument('--maxcount', type=float, default=1e6, help='max number of records in each shard (set big if you want a single shard)')
parser.add_argument('--shards', default='', help='directory where shards are written')
parser.add_argument('--data-dirs', nargs='+', help='list of paths to datasets')
parser.add_argument('--dataset-name', default='sharded-dataset', help='name of sharded dataset')

args = parser.parse_args()
print(args)

# assert args.maxsize > 1e8
# assert args.maxcount < 1e10

def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()

all_keys = set()

def write_dataset(imgdirs, imgname, base=""):

    # We're using a multi-folder generalization of the torchvision ImageFolder class to parse the metadata; 
    # however, we will read the compressed images directly from disk (to avoid having to reencode them)
    ds = MultirootImageFolder(imgdirs, 1.0)
    nimages = len(ds.imgs)
    print("# of images", nimages)

    # for key in list(ds.class_to_idx.keys()):
    #     print(key, ds.class_to_idx[key])

    # We shuffle the indexes to make sure that we don't get any large sequences of a single class in the dataset.
    indexes = list(range(nimages))
    random.shuffle(indexes)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(base, f"{imgname}_%06d.tar")

    counter = 0

    with open('places365_val.txt') as f:
        targets = f.readlines()
    
    with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
        for i in indexes:

            # Internal information from the image dataset instance: the file name and the numerical class.
            fname, _ = ds.imgs[i]
            cls = int(targets[i].split()[-1])

            # Read the JPEG-compressed image file contents.
            image = readfile(fname)

            # Construct a unique key from the filename.
            key = os.path.splitext(os.path.basename(fname))[0]

            # Useful check.
            # assert key not in all_keys  # for some reason this gives an error sometimes
            all_keys.add(key)

            # Construct a sample.
            xkey = key if args.filekey else "%07d" % i
            sample = {"__key__": xkey, "jpg": image, "cls": cls}

            # Write the sample to the sharded tar archives.
            sink.write(sample)

            # Perhaps print out a bunch of useful stuff:
            print("counter, i, fname, cls", counter, i, fname, cls)      

            counter += 1              

write_dataset(args.data_dirs, args.dataset_name, base=args.shards)