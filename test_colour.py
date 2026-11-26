import unittest
import cv2
import numpy as np
from colour_detector import detect, EXPECTED
from engine import Tracker, procedure

class DetectionTests(unittest.TestCase):
    def picture(self, colours, gap=0):
        image=np.full((400,500,3),65,np.uint8)
        paints={'red':(0,0,220),'blue':(220,0,0),'yellow':(0,220,220),'green':(0,180,0)}
        for i,c in enumerate(colours):
            x=40+i*(60+gap)
            cv2.rectangle(image,(x,160),(x+59,219),paints[c],-1)
        return image
    def test_states_and_rotation(self):
        for n in range(1,5):
            for k in range(4):
                self.assertEqual(detect(np.ascontiguousarray(np.rot90(self.picture(EXPECTED[:n]),k)))['step'],'S'+str(n))
    def test_incorrect_order(self):
        r=detect(self.picture(['red','blue','green','yellow']))
        self.assertEqual(r['arrangement'],'incorrect_visible_order')
        self.assertEqual(r['step'],'unknown')
    def test_gaps(self):
        self.assertEqual(detect(self.picture(EXPECTED,gap=30))['step'],'unknown')
    def test_empty(self):
        self.assertEqual(detect(self.picture([]))['step'],'unknown')
    def test_duplicate(self):
        self.assertEqual(detect(self.picture(['red','blue','red']))['step'],'unknown')
    def test_tracker_does_not_invent_steps(self):
        tracker=Tracker(procedure(),stable=3)
        for t in [0,.5,1]:tracker.update(t,'S4')
        self.assertEqual(set(tracker.seen),{'S4'})
        self.assertTrue(tracker.events)
if __name__=='__main__':unittest.main()
