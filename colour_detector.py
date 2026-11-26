
import cv2
import numpy as np

EXPECTED = ['red', 'blue', 'yellow', 'green']
RANGES = {'red': [(0, 10), (165, 179)], 'blue': [(90, 135)],
          'yellow': [(25, 38)], 'green': [(39, 88)]}
DRAW = {'red': (40,40,255), 'blue': (255,100,50),
        'yellow': (0,255,255), 'green': (0,220,0)}

def detect(frame):
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame,(5,5),0), cv2.COLOR_BGR2HSV)
    found = []
    for colour, ranges in RANGES.items():
        mask = np.zeros((height,width), np.uint8)
        for lo,hi in ranges:
            mask |= cv2.inRange(hsv, (lo,160 if colour=="red" else 110,20), (hi,255,255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < max(100,width*height*.002) or area > width*height*.25: continue
            rect = cv2.minAreaRect(c); a,b = rect[1]
            if min(a,b)<8 or max(a,b)/min(a,b)>4: continue
            if area/(a*b+1e-6)<.64: continue
            x,y,w,h = cv2.boundingRect(c)
            if x<=1 or y<=1 or x+w>=width-1 or y+h>=height-1: continue
            box = cv2.boxPoints(rect)
            found.append({'colour':colour,'center':np.array(rect[0]),'corners':box,
                          'area':float(area),'bbox':[int(x),int(y),int(w),int(h)]})
    counts = {c:sum(d['colour']==c for d in found) for c in EXPECTED}
    result = {'step':'unknown','arrangement':'uncertain','reason':'No clear red base in view.',
              'colour_order':[], 'detections':[{'colour':d['colour'],'bbox':d['bbox']} for d in found]}
    if counts['red']!=1: return result
    if any(n>1 for n in counts.values()):
        result['reason']='Multiple regions of one colour. Isolate one build or adjust the work area.'; return result
    if len(found)==1:
        result.update(step='S1',arrangement='valid_visible_state',reason='Red region detected.',colour_order=['red'])
        return result
    centres = np.array([d['center'] for d in found])
    _,_,vh = np.linalg.svd(centres-centres.mean(axis=0),full_matrices=False)
    axis=vh[0]; projection=centres@axis
    idx=np.argsort(projection)
    if found[int(idx[-1])]['colour']=='red': idx=idx[::-1];axis=-axis
    ordered=[found[int(i)] for i in idx]
    order=[d['colour'] for d in ordered]
    result['colour_order']=order
    normal=np.array([-axis[1],axis[0]])
    cross=centres@normal
    widths=[float(np.ptp(d['corners']@normal)) for d in ordered]
    if np.ptp(cross)>min(widths)*.40:
        result['reason']='Regions are not aligned as one straight build.';return result
    for first,second in zip(ordered,ordered[1:]):
        gap=float(np.min(second['corners']@axis)-np.max(first['corners']@axis))
        size=min(np.ptp(first['corners']@axis),np.ptp(second['corners']@axis))
        if gap>max(5,size*.32) or gap < -size*.40:
            result['reason']='Regions are separated or overlap too much to verify.';return result
    if order != EXPECTED[:len(order)]:
        result.update(arrangement='incorrect_visible_order',reason='Visible colours do not match red → blue → yellow → green.')
        return result
    result.update(step='S'+str(len(order)),arrangement='valid_visible_state',reason='Aligned colour regions match the expected order.')
    return result

def annotate(frame, result):
    image=frame.copy()
    for d in result['detections']:
        x,y,w,h=d['bbox'];colour=DRAW[d['colour']]
        cv2.rectangle(image,(x,y),(x+w,y+h),colour,2)
        cv2.putText(image,d['colour'],(x,max(15,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,colour,2)
    return image
