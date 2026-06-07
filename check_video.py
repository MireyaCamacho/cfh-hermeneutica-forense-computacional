import cv2 
cap = cv2.VideoCapture("corpus_c/costa_caribe/Caso 03\uff5c Audiencia de Reconocimiento Subcaso Costa Caribe \uff5c 18 de julio de 2022.mp4") 
fps = cap.get(cv2.CAP_PROP_FPS) 
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
print(f"FPS={fps:.1f}, Frames={total}, Dur={total/fps/3600:.1f}h") 
import numpy as np 
for t in [1000,3000,7000,10000,15000,20000,25000]: 
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t*fps)); ret,f=cap.read(); print(f"t={t}s: {'OK '+str(f.shape) if ret else 'SIN FRAME'}") 
cap.release() 
