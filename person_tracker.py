import cv2
import numpy as np
import time

known_hashes = []

NEW_PERSON_HAMMING_THRESHOLD = 14
COOLDOWN_SECONDS = 3.0
_last_hello_time = 0.0

def _dhash(face_bgr, hash_size=8):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten().astype(np.uint8)

def _hamming(a, b):
    return int(np.sum(a != b))

def is_new_person(face_bgr):
    global _last_hello_time

    now = time.time()
    if now - _last_hello_time < COOLDOWN_SECONDS:
        return False

    h = _dhash(face_bgr)

    if not known_hashes:
        known_hashes.append(h)
        _last_hello_time = now
        return True

    dists = [_hamming(h, k) for k in known_hashes]
    best = min(dists)

    if best > NEW_PERSON_HAMMING_THRESHOLD:
        known_hashes.append(h)
        _last_hello_time = now
        return True

    return False
