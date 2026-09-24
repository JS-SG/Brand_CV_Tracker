"""Split annotated State Farm raw images/labels into YOLO train/val folders."""
from __future__ import annotations
import argparse, random, shutil
from pathlib import Path


def prepare(raw_images, raw_labels, out_root, val_count=1, seed=42):
    raw_images=Path(raw_images); raw_labels=Path(raw_labels); out=Path(out_root)
    images=sorted([p for p in raw_images.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
    if not images: raise RuntimeError(f'No images in {raw_images}')
    random.seed(seed); random.shuffle(images)
    val=set(p.name for p in images[:max(1,min(val_count,len(images)-1))]) if len(images)>1 else set()
    for split in ('train','val'):
        (out/'images'/split).mkdir(parents=True,exist_ok=True); (out/'labels'/split).mkdir(parents=True,exist_ok=True)
    for img in images:
        split='val' if img.name in val else 'train'
        shutil.copy2(img,out/'images'/split/img.name)
        label=raw_labels/f'{img.stem}.txt'
        if not label.exists(): raise FileNotFoundError(f'Missing label: {label}')
        shutil.copy2(label,out/'labels'/split/label.name)
    print(f'Prepared {len(images)} images: train={len(images)-len(val)}, val={len(val)}')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--images',default='dataset_statefarm/raw'); p.add_argument('--labels',default='dataset_statefarm/labels/raw'); p.add_argument('--output',default='dataset_statefarm'); p.add_argument('--val-count',type=int,default=1); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); prepare(a.images,a.labels,a.output,a.val_count,a.seed)
if __name__=='__main__': main()
