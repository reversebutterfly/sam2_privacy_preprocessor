"""
Patch sam2_util.py on server to add J&F metrics.
Run on server: python patch_sam2_util_jf.py
"""
import sys

TARGET = '/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2/sam2_util.py'

with open(TARGET, 'r') as f:
    src = f.read().replace('\r\n', '\n').replace('\r', '\n')

orig = src  # keep for diff

# ── 1. Add import ─────────────────────────────────────────────────────────────
OLD = 'from sam2.build_sam import build_sam2_video_predictor'
NEW = ('from sam2.build_sam import build_sam2_video_predictor\n'
       'from metrics_jf import jf_score')
assert OLD in src, "import anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 2. Add J/F accumulators after existing iou_count init ─────────────────────
OLD = (
    '    seed_everything(seed=args.seed)\n'
    '    total_iou = 0\n'
    '    iou_count = 0\n'
    '\n'
    '    if skipped_frames is None:\n'
    '        skipped_frames = set()'
)
NEW = (
    '    seed_everything(seed=args.seed)\n'
    '    total_iou = 0\n'
    '    iou_count = 0\n'
    '    total_j = 0.0\n'
    '    total_f = 0.0\n'
    '    total_jf = 0.0\n'
    '    jf_count = 0\n'
    '\n'
    '    if skipped_frames is None:\n'
    '        skipped_frames = set()'
)
assert OLD in src, "accumulator anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 3. Compute jf per frame, right after iou_count += 1 ──────────────────────
OLD = (
    '                total_iou += iou_img\n'
    '                iou_count += 1\n'
    '                print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")\n'
    '                print(f"Current iou_count: {iou_count}")'
)
NEW = (
    '                total_iou += iou_img\n'
    '                iou_count += 1\n'
    '                print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")\n'
    '                print(f"Current iou_count: {iou_count}")\n'
    '                try:\n'
    '                    mask_pred_sq = np.array(out_mask).squeeze()\n'
    '                    mask_gt_sq   = np.array(mask_gt).squeeze()\n'
    '                    jf, j, f = jf_score(mask_pred_sq, mask_gt_sq)\n'
    '                    total_j  += j\n'
    '                    total_f  += f\n'
    '                    total_jf += jf\n'
    '                    jf_count += 1\n'
    '                except Exception as _e:\n'
    '                    print(f"[jf] skip frame: {_e}")'
)
assert OLD in src, "per-frame iou anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 4. Fix matplotlib leak: close fig unconditionally ─────────────────────────
OLD = (
    '            if args.save_img_with_mask:\n'
    '                os.makedirs(output_dir, exist_ok=True)\n'
    '                save_prefix = "clean" if category == "clean" else "adv"\n'
    '                save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")\n'
    '                plt.savefig(save_path)\n'
    '                plt.close(fig)'
)
NEW = (
    '            if args.save_img_with_mask:\n'
    '                os.makedirs(output_dir, exist_ok=True)\n'
    '                save_prefix = "clean" if category == "clean" else "adv"\n'
    '                save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")\n'
    '                plt.savefig(save_path)\n'
    '            plt.close(fig)'
)
assert OLD in src, "matplotlib close anchor not found"
src = src.replace(OLD, NEW, 1)

# ── 5. Update return statement ────────────────────────────────────────────────
OLD = (
    '    avg_iou = total_iou / iou_count if iou_count > 0 else 0\n'
    '\n'
    '    return avg_iou, iou_count, skipped_frames'
)
NEW = (
    '    avg_iou = total_iou / iou_count if iou_count > 0 else 0\n'
    '    avg_j   = total_j  / jf_count  if jf_count  > 0 else 0.0\n'
    '    avg_f   = total_f  / jf_count  if jf_count  > 0 else 0.0\n'
    '    avg_jf  = total_jf / jf_count  if jf_count  > 0 else 0.0\n'
    '\n'
    '    return avg_iou, iou_count, skipped_frames, avg_jf, avg_j, avg_f'
)
assert OLD in src, "return anchor not found"
src = src.replace(OLD, NEW, 1)

with open(TARGET, 'w') as f:
    f.write(src)

print("sam2_util.py patched OK")
print(f"  import:      {'from metrics_jf import jf_score' in src}")
print(f"  accumulators: {'total_jf' in src}")
print(f"  per-frame jf: {'jf_score(mask_pred_sq' in src}")
print(f"  plt.close:    {'plt.close(fig)' in src}")
print(f"  return:       {'avg_jf, avg_j, avg_f' in src}")
