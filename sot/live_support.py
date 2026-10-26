"""Bounded, cancellable frame source and lightweight process telemetry.

Real-time rule: a live camera (or a file paced with --source-fps) never queues old frames.
If processing is slower than the source, older frames are dropped and only the latest
one is handed on, so latency stays bounded instead of growing. Unpaced file playback is
the throughput benchmark and keeps every frame.
"""
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
import cv2
import psutil

REDUCED = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
           8: cv2.IMREAD_REDUCED_COLOR_8}


def is_live(source):
    s = str(source)
    return s.isdigit() or s.startswith(('/dev/video', 'rtsp://', 'http://', 'https://')) or '!' in s


def reduce_factor(width, work_width):
    """Largest 2^k (<=8) that keeps the decoded width >= work_width (JPEG DCT downscaling is ~free)."""
    f = 1
    while work_width and f < 8 and width / (f * 2) >= work_width:
        f *= 2
    return f


def shrink(image, work_width):
    if work_width and image.shape[1] > work_width:
        s = work_width / image.shape[1]
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return image


class Frame(dict):
    """image, prepared, decode_ms, pre_ms, t (source seconds, for dt-aware trackers),
    captured_at (perf_counter when the frame became available), index, dropped (so far)."""
    __getattr__ = dict.__getitem__


class PrefetchFrames:
    """One reader thread per source. JPEG folders are decoded already reduced (2/4/8x) so every
    later stage (detector letterbox, trackers, CMC, display) works on a small image; video and
    camera frames are shrunk once to work_width. Returned boxes are in that working image."""

    def __init__(self, source, preprocess=None, start_frame=1, source_fps=0, work_width=1280):
        self.source = str(source)
        self.start_frame = start_frame
        self.source_fps = source_fps
        self.preprocess = preprocess
        self.work_width = work_width
        self.live = is_live(source)
        self.drop = self.live or source_fps > 0  # real-time: keep only the newest frame
        self.queue = Queue(1)
        self.dropped = 0
        self.cancelled = Event()
        self.thread = Thread(target=self.read, daemon=True)
        self.thread.start()

    def put(self, value):
        if self.drop and isinstance(value, Frame):
            while not self.cancelled.is_set():
                try:
                    self.queue.put_nowait(value)
                    return
                except Full:
                    try:
                        self.queue.get_nowait()  # newest wins; the consumer never sees stale frames
                        self.dropped += 1
                    except Empty:
                        pass
            return
        while not self.cancelled.is_set():
            try:
                self.queue.put(value, timeout=.1)
                return
            except Full:
                pass

    def emit(self, image, start, t, index):
        decoded = time.perf_counter()
        prepared = self.preprocess(image) if self.preprocess is not None else None
        end = time.perf_counter()
        self.put(Frame(image=image, prepared=prepared, decode_ms=(decoded-start)*1000, pre_ms=(end-decoded)*1000,
                       t=t, captured_at=decoded, index=index, dropped=self.dropped))

    def read(self):
        try:
            if Path(self.source).is_dir():
                self.read_folder()
            else:
                self.read_capture()
            self.put(None)
        except Exception as error:
            self.put(error)

    def read_folder(self):
        files = sorted(Path(self.source).glob('*.jpg'))
        if not files:
            raise ValueError(f'No .jpg frames in {self.source}')
        probe = cv2.imread(str(files[0]))
        flag = REDUCED[reduce_factor(probe.shape[1], self.work_width)]
        shape = None
        fps = self.source_fps or 30.
        origin = time.perf_counter()
        i = self.start_frame - 1
        while not self.cancelled.is_set() and i < len(files):
            if self.source_fps > 0:
                # Simulated camera: frame i exists at origin + i/fps. Skip frames whose time passed.
                due = origin + (i - self.start_frame + 1) / self.source_fps
                now = time.perf_counter()
                if now < due:
                    self.cancelled.wait(due - now)
                else:
                    behind = int((now - due) * self.source_fps)
                    if behind:
                        self.dropped += behind
                        i += behind
                        if i >= len(files):
                            break
            start = time.perf_counter()
            image = cv2.imread(str(files[i]), flag)
            if image is None:
                raise ValueError(f'Cannot decode {files[i]}')
            image = shrink(image, self.work_width)
            if shape is None:
                shape = image.shape[:2]
            elif image.shape[:2] != shape:  # the HF mirror has a few odd-sized frames
                image = cv2.resize(image, shape[::-1])
            self.emit(image, start, i / fps, i + 1)
            i += 1

    def read_capture(self):
        cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        if not cap.isOpened():
            raise ValueError(f'Cannot open {self.source}')
        try:
            if self.live:
                # Ask the camera for a working-size stream instead of resizing every frame.
                if self.work_width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.work_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, round(self.work_width * 9 / 16))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.
            for _ in range(self.start_frame - 1):
                cap.grab()
            origin = time.perf_counter()
            index = self.start_frame - 1
            while not self.cancelled.is_set():
                if not self.live and self.source_fps > 0:
                    due = origin + (index - self.start_frame + 1) / self.source_fps
                    now = time.perf_counter()
                    if now < due:
                        self.cancelled.wait(due - now)
                    else:
                        for _ in range(int((now - due) * self.source_fps)):  # grab() skips without decoding
                            if not cap.grab():
                                return
                            index += 1
                            self.dropped += 1
                start = time.perf_counter()
                ok, image = cap.read()
                if not ok:
                    if self.live:
                        raise ValueError(f'Camera {self.source} stopped delivering frames')
                    return
                t = (start - origin) if self.live else index / (self.source_fps or fps)
                self.emit(shrink(image, self.work_width), start, t, index + 1)
                index += 1
        finally:
            cap.release()

    def __next__(self):
        while not self.cancelled.is_set():
            try: item = self.queue.get(timeout=.1)
            except Empty: continue
            if item is None: raise StopIteration
            if isinstance(item, Exception): raise item
            return item
        raise StopIteration

    def close(self):
        self.cancelled.set()
        self.thread.join(timeout=2)


class ResourceSampler:
    def __init__(self):
        self.stop = Event()
        self.process = psutil.Process()
        self.latest = {}
        self.thread = Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        self.process.cpu_percent()
        psutil.cpu_percent(percpu=True)
        while not self.stop.wait(.5):
            cores = psutil.cpu_percent(percpu=True)
            self.latest = dict(sampled_at_monotonic=time.monotonic(),process_cpu_one_core_pct=self.process.cpu_percent(),
                system_cpu_per_core_pct=cores,
                system_cpu_total_pct=sum(cores)/max(1,len(cores)),
                process_rss_mb=self.process.memory_info().rss/2**20,
                system_used_ram_mb=psutil.virtual_memory().used/2**20,
                system_available_ram_mb=psutil.virtual_memory().available/2**20,
                cpu_cores=psutil.cpu_count(),
                system_idle_pct=100-sum(cores)/max(1,len(cores)))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


class InferenceFrames:
    """Ordered, bounded detector stage. A frame keeps its own detections.
    Wraps a PrefetchFrames without owning it, so switching tracker type never reopens a camera."""
    def __init__(self, source, detector):
        self.source = source
        self.detector = detector
        self.queue = Queue(1)
        self.cancelled = Event()
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    @property
    def dropped(self):
        return self.source.dropped

    def put(self, value):
        while not self.cancelled.is_set():
            try:
                self.queue.put(value, timeout=.05)
                return
            except Full: pass

    def run(self):
        try:
            while not self.cancelled.is_set():
                try: item = next(self.source)
                except StopIteration: break
                detected, timing = self.detector(item.image, item.prepared)
                item = Frame(item, detections=detected, timing=timing)
                self.put(item)
            self.put(None)
        except Exception as error:
            self.put(error)

    def __next__(self):
        while not self.cancelled.is_set():
            try: item = self.queue.get(timeout=.05)
            except Empty: continue
            if item is None: raise StopIteration
            if isinstance(item, Exception): raise item
            return item
        raise StopIteration

    def close(self, close_source=True):
        self.cancelled.set()
        if close_source:
            self.source.close()
        self.thread.join(timeout=5)  # run() polls cancelled between frames, so it exits on its own
