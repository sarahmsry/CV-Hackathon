import sys
import cv2
import numpy as np
from imutils import contours as imcontours, perspective

def measure_height(image_path, ref_height_mm=None):
    img = cv2.imread(image_path)

    if ref_height_mm is None:
        try:
            ref_height_mm = float(input('Reference height (mm): '))
        except ValueError:
            print('Invalid height.'); return

    ref_roi = cv2.selectROI('Reference', img, False, True)
    cv2.destroyWindow('Reference')
    if ref_roi[3] == 0:
        print('No reference selected'); return

    scale = ref_roi[3] / ref_height_mm

    person_roi = cv2.selectROI('Person', img, False, True)
    cv2.destroyWindow('Person')
    if person_roi[3] == 0:
        print('No person selected'); return

    height_cm = (person_roi[3] / scale) / 10
    print(f"Estimated height: {height_cm:.1f} cm")

    out = img.copy()
    x,y,w,h = person_roi
    cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.putText(out,f"{height_cm:.1f}cm",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    cv2.imshow('Result',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = sys.argv[1] if len(sys.argv)>1 else input('Image path: ').strip()
    ref = float(sys.argv[2]) if len(sys.argv)>2 else None
    measure_height(img, ref)
