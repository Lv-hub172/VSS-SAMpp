import os
import numpy as np
import nibabel as nib
import pickle
import random
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom


CONFIG = {
    "base_dir": None,

    "label_dir": None,
    "image_dir": None,

    "images_va": "imageval",
    "labels_va": "labelsval",

    "save_2d_dir": None,

    "label_mapping": {
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0,
        14: 9,
        15: 10
    },


    "HU_min": -200,
    "HU_max": 300,
}

def remap_labels(cfg):
    label_dir = cfg["label_dir"]
    mapping = cfg["label_mapping"]

    print(f"Total labels: {len(os.listdir(label_dir))}")

    for filename in os.listdir(label_dir):
        if not filename.endswith(".nii.gz"):
            continue

        label_path = os.path.join(label_dir, filename)

        nii = nib.load(label_path)
        data = nii.get_fdata().astype(np.int32)

        new_data = data.copy()

        for old, new in mapping.items():
            new_data[data == old] = new

        nib.save(
            nib.Nifti1Image(new_data, nii.affine, nii.header),
            label_path
        )

        print(
            f"[OK] {filename} | "
            f"before={np.unique(data)} -> after={np.unique(new_data)}"
        )



def get_all_5slice(cfg):
    base_dir = cfg["base_dir"]
    save_dir = cfg["save_2d_dir"]

    images_dir = os.path.join(base_dir, cfg["images_va"])
    labels_dir = os.path.join(base_dir, cfg["labels_va"])

    os.makedirs(save_dir, exist_ok=True)

    case_list = sorted([f for f in os.listdir(images_dir) if f.startswith("AMOS")])

    for case in tqdm(case_list):
        case_id = case[5:9]
        case_dir = os.path.join(save_dir, case_id)

        os.makedirs(case_dir + "/images", exist_ok=True)
        os.makedirs(case_dir + "/masks", exist_ok=True)

        img = nib.load(os.path.join(images_dir, case)).get_fdata()
        mask = nib.load(os.path.join(labels_dir, f"AMOS_{case_id}.nii.gz")).get_fdata()

        h, w = img.shape[:2]
        if h != 256 or w != 256:
            img = zoom(img, (256 / h, 256 / w, 1.0), order=3)
            mask = zoom(mask, (256 / h, 256 / w, 1.0), order=0)

        img = np.concatenate([img[..., :1]] * 2 + [img] + [img[..., -1:]] * 2, axis=-1)
        mask = np.concatenate([mask[..., :1]] * 2 + [mask] + [mask[..., -1:]] * 2, axis=-1)

        for i in range(2, img.shape[2] - 2):
            img_2d = np.flip(np.rot90(img[:, :, i-2:i+3], 1), axis=1)
            mask_2d = np.flip(np.rot90(mask[:, :, i-2:i+3], 1), axis=1)

            with open(f"{case_dir}/images/2Dimage_{i-2:04d}.pkl", "wb") as f:
                pickle.dump(img_2d, f)

            with open(f"{case_dir}/masks/2Dmask_{i-2:04d}.pkl", "wb") as f:
                pickle.dump(mask_2d, f)



def get_csv(cfg):
    save_dir = cfg["save_2d_dir"]
    images_va_dir = os.path.join(cfg["base_dir"], cfg["images_va"])

    train_csv = os.path.join(save_dir, "training.csv")
    test_csv = os.path.join(save_dir, "test.csv")

    all_cases = [d for d in os.listdir(save_dir) if d.isdigit()]
    random.shuffle(all_cases)

    test_cases = [f[5:9] for f in os.listdir(images_va_dir)]

    train_cases = [c for c in all_cases if c not in test_cases]

    def build_df(case_list):
        paths = []
        for c in case_list:
            img_dir = os.path.join(save_dir, c, "images")
            imgs = os.listdir(img_dir)

            for i in imgs:
                paths.append(f"{c}/images/{i}")

        df = pd.DataFrame(paths, columns=["image_pth"])
        df["mask_pth"] = df["image_pth"].str.replace(
            "/images/2Dimage_", "/masks/2Dmask_"
        )

        df = df.sample(frac=1).reset_index(drop=True)

        return df

    build_df(train_cases).to_csv(train_csv, index=False)
    build_df(test_cases).to_csv(test_csv, index=False)



def get_data_statistics(cfg):
    img_dir =None
    HU_min, HU_max = cfg["HU_min"], cfg["HU_max"]

    means, sqs, counts = [], [], []

    for f in tqdm(sorted(os.listdir(img_dir))):
        if not f.startswith("AMOS"):
            continue

        img = nib.load(os.path.join(img_dir, f)).get_fdata()
        img = np.clip(img, HU_min, HU_max)
        img = (img - HU_min) / (HU_max - HU_min) * 255.0

        means.append(np.sum(img))
        sqs.append(np.sum(img ** 2))
        counts.append(img.size)

    mean = sum(means) / sum(counts)
    std = np.sqrt(sum(sqs) / sum(counts) - mean ** 2)

    print("Global Mean:", mean)
    print("Global Std:", std)



if __name__ == "__main__":
    remap_labels(CONFIG)
    get_all_5slice(CONFIG)
    get_csv(CONFIG)
    get_data_statistics(CONFIG)



