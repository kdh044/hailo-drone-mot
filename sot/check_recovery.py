"""Check state transitions with an injected tracker failure, not tracking accuracy."""
import numpy as np
from benchmark_system import Engine

class Failed:
    def update(self,image):
        return False,(0,0,0,0)

def main():
    image=np.zeros((240,320,3),np.uint8)
    image[60:90,90:110]=np.random.default_rng(8).integers(0,255,(30,20,3),dtype=np.uint8)
    dets=np.array([[90,60,110,90,.9,0.]])
    empty=np.empty((0,6))
    for method in ('KCF','CSRT'):
        e=Engine(method,recovery_frames=2)
        assert e.update(image,dets)['next_mode']=='TRACK'
        e.tracker=Failed()
        assert e.update(image,empty)['next_mode']=='RECOVER'
        assert e.update(image,empty)['next_mode']=='TRACK'
        e.tracker=Failed()
        assert e.update(image,empty)['next_mode']=='RECOVER'
        blank=np.zeros_like(image)
        assert e.update(blank,empty)['next_mode']=='RECOVER'
        assert e.update(blank,empty)['next_mode']=='DETECT'
        assert e.update(image,dets)['next_mode']=='TRACK'
        e.select_target([90,60,20,30])
        assert e.state=='DETECT' and e.track_id is None and e.tracker is None
        assert e.update(image,dets)['next_mode']=='TRACK'
        print(method,'PASS: select, track, recover, timeout, redetect, change target')
if __name__=='__main__':main()
