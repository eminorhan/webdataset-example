import os
import os.path
import random
import math
import argparse
import torch
import webdataset as wds
from multiroot_image_folder import MultirootImageFolder

parser = argparse.ArgumentParser('Generate sharded dataset from a list of image directories')
parser.add_argument('--filekey', action='store_true', help='use file as key (default: index)')
parser.add_argument('--maxsize', type=float, default=1e16, help='max size of each shard (set big if you want a single shard)')
parser.add_argument('--maxcount', type=float, default=1e6, help='max number of records in each shard (set big if you want a single shard)')
parser.add_argument('--data-fraction', type=float, default=1.0, help='fraction of data to use')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--shards', default='', help='directory where shards are written')
parser.add_argument('--data-dirs', nargs='+', help='list of paths to datasets')
parser.add_argument('--dataset-name', default='sharded-dataset', help='name of sharded dataset')
parser.add_argument('--cache-path', default='sharded-dataset.pth', help='name of sharded dataset')

args = parser.parse_args()
print(args)

random.seed(args.seed)

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
    if args.cache_path and os.path.exists(args.cache_path):
        print("Loading training dataset from {}".format(args.cache_path))
        ds = torch.load(args.cache_path)
    else:
        print("Building training dataset from scratch")
        ds = MultirootImageFolder(imgdirs, 1.0)
        torch.save(ds, args.cache_path)

    nimages = len(ds.imgs)
    print("# of all images", nimages)

    # for key in list(ds.class_to_idx.keys()):
    #     print(key, ds.class_to_idx[key])

    indexes = list(range(nimages))

    # keep a certain fraction of the images
    if args.data_fraction < 1.0:
        num_images = math.ceil(nimages * args.data_fraction)
        start_ind = random.randint(0, nimages-num_images-1)
        print('Image start index:', start_ind)
        indexes = indexes[start_ind:(start_ind+num_images)]
        print("# of images kept", len(indexes))

    random.shuffle(indexes)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(base, f"{imgname}_{args.data_fraction}_{args.seed}_%06d.tar")

    counter = 0

    with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
        for i in indexes:

            # Internal information from the image dataset instance: the file name and the numerical class.
            fname, cls = ds.imgs[i]
            assert cls == ds.targets[i]

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
            if counter % 10000 == 0:
                print("counter, i, fname, cls", counter, i, fname, cls)      

            counter += 1              

write_dataset(args.data_dirs, args.dataset_name, base=args.shards)