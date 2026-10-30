"""Generate COCO-C style corrupted validation sets (5 corruptions x severity 5)."""
import os
import time
from multiprocessing import Pool

import imagecorruptions
import numpy as np
from PIL import Image

SRC = "/root/autodl-tmp/project_marn-detr/datasets/coco/val2017"
BASE = "/root/autodl-tmp/datasets/coco_c"
ORIG = "/root/autodl-tmp/project_marn-detr/datasets/coco"
CORRS = ["gaussian_noise", "motion_blur", "contrast", "jpeg_compression", "saturate"]
SEV = 5


def setup():
    for name in CORRS:
        d = os.path.join(BASE, name)
        os.makedirs(os.path.join(d, "val2017"), exist_ok=True)
        for link, target in (("train2017", os.path.join(ORIG, "train2017")),
                             ("annotations", os.path.join(ORIG, "annotations"))):
            p = os.path.join(d, link)
            if not os.path.exists(p):
                os.symlink(target, p)


def work(job):
    name, fname = job
    dst = os.path.join(BASE, name, "val2017", fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 0
    arr = np.array(Image.open(os.path.join(SRC, fname)).convert("RGB"))
    out = imagecorruptions.corrupt(arr, corruption_name=name, severity=SEV)
    Image.fromarray(out).save(dst, quality=95)
    return 1


if __name__ == "__main__":
    setup()
    files = sorted(os.listdir(SRC))
    jobs = [(n, f) for n in CORRS for f in files]
    print("total jobs", len(jobs), flush=True)
    t0 = time.time()
    done = 0
    with Pool(4) as pool:
        for i, n in enumerate(pool.imap_unordered(work, jobs, chunksize=20), 1):
            done += n
            if i % 2000 == 0:
                el = time.time() - t0
                print(f"{i}/{len(jobs)} new={done} elapsed={el:.0f}s eta={el / i * (len(jobs) - i):.0f}s",
                      flush=True)
    print("DONE", done, "in", round(time.time() - t0), "s", flush=True)
