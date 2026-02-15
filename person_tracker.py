import cv2
import numpy as np
import time

# ----- TUNING -----
NEW_PERSON_HAMMING_THRESHOLD = 12   # higher = less "new" triggers (start 12-16)
STABLE_FRAMES_REQUIRED = 8          # face must be stable for this many frames
SAME_PERSON_COOLDOWN_SEC = 25       # don't greet same person again within this time
# -------------------

known_hashes = []          # list of hashes for each known person
known_last_seen = []       # last-seen time for each known person index

_last_seen_hash = None
_stable_count = 0


def _preprocess(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def _dhash(gray, hash_size=8):
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten().astype(np.uint8)


def _hamming(a, b):
    return int(np.sum(a != b))


def _match_person(h):
    """
    Returns (best_index, best_distance) among known persons.
    If no known persons, returns (-1, large).
    """
    if not known_hashes:
        return -1, 999

    dists = [_hamming(h, k) for k in known_hashes]
    best_idx = int(np.argmin(dists))
    return best_idx, dists[best_idx]


def should_greet(face_bgr):
    global _last_seen_hash, _stable_count

    # Ignore very small face crops (likely passers-by / far away)
    h_img, w_img = face_bgr.shape[:2]
    if w_img < 80 or h_img < 80:
        return False

    now = time.time()
    gray = _preprocess(face_bgr)
    h = _dhash(gray)

    # ---------- stability gate ----------
    if _last_seen_hash is None:
        _last_seen_hash = h
        _stable_count = 1
        return False

    if _hamming(h, _last_seen_hash) <= 6:
        _stable_count += 1
    else:
        _stable_count = 1
        _last_seen_hash = h

    if _stable_count < STABLE_FRAMES_REQUIRED:
        return False
    # -----------------------------------

    idx, dist = _match_person(h)

    # Case A: New person
    if idx == -1 or dist > NEW_PERSON_HAMMING_THRESHOLD:
        known_hashes.append(h)
        known_last_seen.append(now)
        _last_seen_hash = None
        _stable_count = 0
        return True

    # Case B: Known person but cooldown passed
    if now - known_last_seen[idx] > SAME_PERSON_COOLDOWN_SEC:
        known_last_seen[idx] = now
        _last_seen_hash = None
        _stable_count = 0
        return True

    known_last_seen[idx] = now
    return False
