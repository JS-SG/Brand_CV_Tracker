"""Interactive YOLO annotation tool for the RIGHT-SIDE BASKET State Farm logo."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple
import cv2


class Annotator:
    def __init__(self, images: Path, labels: Path):
        self.images = sorted([p for p in images.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
        self.labels = labels
        self.labels.mkdir(parents=True, exist_ok=True)
        self.i = 0
        self.start: Optional[Tuple[int,int]] = None
        self.end: Optional[Tuple[int,int]] = None
        self.dragging = False
        self.box = None

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True; self.start = (x,y); self.end = (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False; self.end = (x,y)
            if self.start and self.end:
                x1,y1 = self.start; x2,y2 = self.end
                x1,x2 = sorted((x1,x2)); y1,y2 = sorted((y1,y2))
                if x2-x1 >= 5 and y2-y1 >= 5:
                    self.box = (x1,y1,x2,y2)

    def run(self):
        if not self.images:
            print('No images found.')
            return
        win = 'STATEFARM RIGHT BASKET | drag box | s=save | n=negative | c=clear | q=quit'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self.mouse)
        while self.i < len(self.images):
            img_path = self.images[self.i]
            label_path = self.labels / f'{img_path.stem}.txt'
            frame = cv2.imread(str(img_path))
            if frame is None:
                self.i += 1; continue
            h,w = frame.shape[:2]
            self.box = None
            while True:
                view = frame.copy()
                if self.box:
                    x1,y1,x2,y2 = self.box
                    cv2.rectangle(view,(x1,y1),(x2,y2),(0,255,0),3)
                    cv2.putText(view,'STATEFARM_RIGHT_BASKET',(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)
                if self.dragging and self.start and self.end:
                    cv2.rectangle(view,self.start,self.end,(0,165,255),2)
                cv2.putText(view,f'{self.i+1}/{len(self.images)} {img_path.name}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                cv2.imshow(win,view)
                key=cv2.waitKey(20)&0xFF
                if key in (ord('q'),27):
                    cv2.destroyAllWindows(); return
                if key==ord('c'):
                    self.box=None
                elif key in (ord('n'),32):
                    label_path.write_text('',encoding='utf-8'); print('negative:',img_path.name); self.i+=1; break
                elif key==ord('s'):
                    with label_path.open('w',encoding='utf-8') as f:
                        if self.box:
                            x1,y1,x2,y2=self.box
                            xc=((x1+x2)/2)/w; yc=((y1+y2)/2)/h
                            bw=(x2-x1)/w; bh=(y2-y1)/h
                            f.write(f'0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')
                    print('saved:',label_path)
                    self.i+=1; break
        cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--images',default='dataset_statefarm/raw')
    p.add_argument('--labels',default='dataset_statefarm/labels/raw')
    a=p.parse_args(); Annotator(Path(a.images),Path(a.labels)).run()

if __name__=='__main__': main()
