"""
Patch uap_eval_heldout_jpeg.py to receive and report J&F metrics.
Run on server: python patch_eval_jf.py
"""

TARGET = '/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2/uap_eval_heldout_jpeg.py'

with open(TARGET, 'r') as f:
    src = f.read().replace('\r\n', '\n').replace('\r', '\n')

# ── 1. Add J&F accumulators in run() ─────────────────────────────────────────
OLD = (
    '    total_miou_clean = 0.0\n'
    '    total_miou_adv = 0.0\n'
    '    iou_count = 0\n'
    '    iou_count_adv = 0'
)
NEW = (
    '    total_miou_clean = 0.0\n'
    '    total_miou_adv = 0.0\n'
    '    iou_count = 0\n'
    '    iou_count_adv = 0\n'
    '    total_jf_clean = 0.0;  total_j_clean = 0.0;  total_f_clean = 0.0\n'
    '    total_jf_adv   = 0.0;  total_j_adv   = 0.0;  total_f_adv   = 0.0\n'
    '    jf_count_clean = 0;    jf_count_adv = 0'
)
assert OLD in src, "accumulator anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 2. Unpack clean process_videos_test (3 values → 6) ───────────────────────
OLD = (
    '        miou_clean, iou_count_video, skipped_frames_video = process_videos_test('
)
NEW = (
    '        miou_clean, iou_count_video, skipped_frames_video, jf_clean, j_clean, f_clean = process_videos_test('
)
assert OLD in src, "clean unpack anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 3. Unpack adv process_videos_test ────────────────────────────────────────
OLD = (
    '        miou_adv, iou_count_adv_video, _ = process_videos_test('
)
NEW = (
    '        miou_adv, iou_count_adv_video, _, jf_adv, j_adv, f_adv = process_videos_test('
)
assert OLD in src, "adv unpack anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 4. Accumulate J&F alongside mIoU ─────────────────────────────────────────
OLD = (
    '        total_miou_clean += miou_clean * iou_count_video\n'
    '        total_miou_adv += miou_adv * iou_count_adv_video\n'
    '        iou_count += iou_count_video\n'
    '        iou_count_adv += iou_count_adv_video'
)
NEW = (
    '        total_miou_clean += miou_clean * iou_count_video\n'
    '        total_miou_adv += miou_adv * iou_count_adv_video\n'
    '        iou_count += iou_count_video\n'
    '        iou_count_adv += iou_count_adv_video\n'
    '        total_jf_clean += jf_clean * iou_count_video\n'
    '        total_j_clean  += j_clean  * iou_count_video\n'
    '        total_f_clean  += f_clean  * iou_count_video\n'
    '        jf_count_clean += iou_count_video\n'
    '        total_jf_adv += jf_adv * iou_count_adv_video\n'
    '        total_j_adv  += j_adv  * iou_count_adv_video\n'
    '        total_f_adv  += f_adv  * iou_count_adv_video\n'
    '        jf_count_adv += iou_count_adv_video'
)
assert OLD in src, "accumulate anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 5. Compute averages and update return ─────────────────────────────────────
OLD = (
    '    avg_miou_clean = total_miou_clean / iou_count if iou_count > 0 else 0\n'
    '    avg_miou_adv = total_miou_adv / iou_count_adv if iou_count_adv > 0 else 0\n'
    '    print(f"iou_count: {iou_count}")\n'
    '\n'
    '    return (video_count, avg_miou_clean, avg_miou_adv,\n'
    '            iou_count, iou_count_adv, total_frame_count,\n'
    '            eval_video_ids, overlap_info)'
)
NEW = (
    '    avg_miou_clean = total_miou_clean / iou_count if iou_count > 0 else 0\n'
    '    avg_miou_adv = total_miou_adv / iou_count_adv if iou_count_adv > 0 else 0\n'
    '    avg_jf_clean = total_jf_clean / jf_count_clean if jf_count_clean > 0 else 0.0\n'
    '    avg_j_clean  = total_j_clean  / jf_count_clean if jf_count_clean > 0 else 0.0\n'
    '    avg_f_clean  = total_f_clean  / jf_count_clean if jf_count_clean > 0 else 0.0\n'
    '    avg_jf_adv   = total_jf_adv   / jf_count_adv   if jf_count_adv   > 0 else 0.0\n'
    '    avg_j_adv    = total_j_adv    / jf_count_adv   if jf_count_adv   > 0 else 0.0\n'
    '    avg_f_adv    = total_f_adv    / jf_count_adv   if jf_count_adv   > 0 else 0.0\n'
    '    print(f"iou_count: {iou_count}")\n'
    '    print(f"[jf] clean: JF={avg_jf_clean*100:.2f}%  J={avg_j_clean*100:.2f}%  F={avg_f_clean*100:.2f}%")\n'
    '    print(f"[jf]   adv: JF={avg_jf_adv*100:.2f}%  J={avg_j_adv*100:.2f}%  F={avg_f_adv*100:.2f}%")\n'
    '\n'
    '    return (video_count, avg_miou_clean, avg_miou_adv,\n'
    '            iou_count, iou_count_adv, total_frame_count,\n'
    '            eval_video_ids, overlap_info,\n'
    '            avg_jf_clean, avg_j_clean, avg_f_clean,\n'
    '            avg_jf_adv,   avg_j_adv,   avg_f_adv)'
)
assert OLD in src, "return anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 6. Unpack the extended return in __main__ ─────────────────────────────────
OLD = (
    '    (video_test, miouimg, miouadv,\n'
    '     frames_clean, frames_adv, frames_train,\n'
    '     eval_video_ids, overlap_info) = run(args, custom_dataset)'
)
NEW = (
    '    (video_test, miouimg, miouadv,\n'
    '     frames_clean, frames_adv, frames_train,\n'
    '     eval_video_ids, overlap_info,\n'
    '     jf_clean, j_clean, f_clean,\n'
    '     jf_adv, j_adv, f_adv) = run(args, custom_dataset)'
)
assert OLD in src, "main unpack anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 7. Add J&F to results JSON ────────────────────────────────────────────────
OLD = (
    '        "miou_clean_pct": round(miouimg * 100, 2),\n'
    '        "miou_adv_pct": round(miouadv * 100, 2),'
)
NEW = (
    '        "miou_clean_pct": round(miouimg * 100, 2),\n'
    '        "miou_adv_pct": round(miouadv * 100, 2),\n'
    '        "jf_clean_pct":  round(jf_clean * 100, 2),\n'
    '        "j_clean_pct":   round(j_clean  * 100, 2),\n'
    '        "f_clean_pct":   round(f_clean  * 100, 2),\n'
    '        "jf_adv_pct":    round(jf_adv   * 100, 2),\n'
    '        "j_adv_pct":     round(j_adv    * 100, 2),\n'
    '        "f_adv_pct":     round(f_adv    * 100, 2),'
)
assert OLD in src, "json anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 8. Add J&F to final print ────────────────────────────────────────────────
OLD = (
    '    print(f":: miouimg: {miouimg * 100:.2f} %, miouadv: {miouadv * 100:.2f} %, "'
)
NEW = (
    '    print(f":: jf_clean: {jf_clean*100:.2f}%  j_clean: {j_clean*100:.2f}%  f_clean: {f_clean*100:.2f}%")\n'
    '    print(f":: jf_adv:   {jf_adv*100:.2f}%  j_adv:   {j_adv*100:.2f}%  f_adv:   {f_adv*100:.2f}%  delta_jf: {(jf_adv-jf_clean)*100:.2f}pp")\n'
    '    print(f":: miouimg: {miouimg * 100:.2f} %, miouadv: {miouadv * 100:.2f} %, "'
)
assert OLD in src, "print anchor not found"
src = src.replace(OLD, NEW, 1)

with open(TARGET, 'w') as f:
    f.write(src)

print("uap_eval_heldout_jpeg.py patched OK")
checks = [
    ('jf accumulators',  'total_jf_clean' in src),
    ('clean unpack',     'jf_clean, j_clean, f_clean = process_videos_test' in src),
    ('adv unpack',       'jf_adv, j_adv, f_adv = process_videos_test' in src),
    ('return extended',  'avg_jf_adv,   avg_j_adv,   avg_f_adv' in src),
    ('main unpack',      'jf_adv, j_adv, f_adv) = run' in src),
    ('json fields',      'jf_adv_pct' in src),
    ('final print',      'delta_jf' in src),
]
for name, ok in checks:
    print(f"  {name}: {ok}")
