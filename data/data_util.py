from torchvision import datasets
import torch
import os

# CUB_DATA_DIR = "YOUR/CUB/DATA/DIR"
# AWA2_DATA_DIR = "YOUR/AWA2/DATA/DIR"
# CELEBA_DATA_DIR = "YOUR/CELEBA/DATA/DIR"
CUB_DATA_DIR = "/home/xxucb/data/pcbm_dataset/CUB"
AWA2_DATA_DIR='/home/xxucb/data/pcbm_dataset/Animals_with_Attributes2'
CELEBA_DATA_DIR = "/home/xxucb/data/pcbm_dataset/celeba"
def get_dataset(args):
    if args.dataset == "cub":
        from .cub import load_cub_data
        from torchvision import transforms
        num_classes = 200
        TRAIN_PKL = os.path.join(CUB_DATA_DIR, "train.pkl")
        VAL_PKL = os.path.join(CUB_DATA_DIR, "val.pkl")
        TEST_PKL = os.path.join(CUB_DATA_DIR, "test.pkl")
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        # print("loading.....")
        train_loader = load_cub_data([TRAIN_PKL], use_attr=True, no_img=False, 
            batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=299, normalizer=normalizer,
            n_classes=num_classes, resampling=True)
        
        val_loader = load_cub_data([VAL_PKL], use_attr=True, no_img=False, 
                batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=299, normalizer=normalizer,
                n_classes=num_classes, resampling=True)

        test_loader = load_cub_data([TEST_PKL], use_attr=True, no_img=False, 
                batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=299, normalizer=normalizer,
                n_classes=num_classes, resampling=True)

        print(len(train_loader.dataset), "training set size")
        print(len(val_loader.dataset), "val set size")
        print(len(test_loader.dataset), "test set size")

    elif args.dataset == "awa2":
        from .awa2 import generate_data
        from torchvision import transforms
        num_classes = 50
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        # print("loading.....")
        train_loader,val_loader,test_loader,_,_=generate_data(args,AWA2_DATA_DIR,resol=299)
        print(len(train_loader.dataset), "training set size")
        print(len(val_loader.dataset), "val set size")
        print(len(test_loader.dataset), "test set size")
    


    elif args.dataset == "celeba":
        from .celeba import generate_data
        from torchvision import transforms
        num_classes = 50
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        # print("loading.....")
        train_loader,val_loader,test_loader,_=generate_data(CELEBA_DATA_DIR)
        print(len(train_loader.dataset), "training set size")
        print(len(val_loader.dataset), "val set size")
        print(len(test_loader.dataset), "test set size")
    
        
    else:
        raise ValueError(args.dataset)

    return train_loader, val_loader, test_loader
