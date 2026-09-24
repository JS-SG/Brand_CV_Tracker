"""Train a separate one-class YOLO model for the State Farm logo on the physical right-side basket."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import torch
from ultralytics import YOLO


def train(data='dataset_statefarm/data.yaml', base_model='yolo11n.pt', epochs=60, batch=8, imgsz=960, device='auto', output='models/statefarm_right_basket.pt', workers=2):
    device = 'cuda:0' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
    model = YOLO(base_model)
    result = model.train(data=str(Path(data).resolve()), epochs=epochs, batch=batch, imgsz=imgsz,
                         device=device, workers=workers, project='runs/detect', name='statefarm_right_basket',
                         exist_ok=True, plots=True, single_cls=True, patience=20, degrees=8, translate=0.05,
                         scale=0.25, fliplr=0.0, mosaic=0.5)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / 'weights' / 'best.pt'
    if not best.exists():
        raise RuntimeError(f'best.pt not found in {save_dir}')
    dst=Path(output); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(best,dst)
    print(f'Saved: {dst.resolve()}')
    try:
        m=model.val(data=str(Path(data).resolve()),device=device)
        print(f'Precision={m.box.mp:.4f} Recall={m.box.mr:.4f} mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f}')
    except Exception as e: print(f'Validation warning: {e}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='dataset_statefarm/data.yaml'); p.add_argument('--base-model',default='yolo11n.pt')
    p.add_argument('--epochs',type=int,default=60); p.add_argument('--batch',type=int,default=8); p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--device',default='auto'); p.add_argument('--output',default='models/statefarm_right_basket.pt'); p.add_argument('--workers',type=int,default=2)
    a=p.parse_args(); train(**vars(a))
if __name__=='__main__': main()
