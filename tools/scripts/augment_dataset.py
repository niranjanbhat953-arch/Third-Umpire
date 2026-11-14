"""
Augment images so every player folder has at least 15 varied, high-quality images.
"""
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance

DATASET = Path("dataset")

for folder in sorted(DATASET.iterdir()):
    if not folder.is_dir() or folder.name.startswith("_"):
        continue
    
    orig_files = [f for f in folder.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"] and not f.name.startswith("aug_")]
    if not orig_files:
        continue
    
    current_count = len(list(folder.glob("*.jpg")))
    target = 18
    
    if current_count < target:
        added = 0
        while len(list(folder.glob("*.jpg"))) < target:
            for f in orig_files:
                if len(list(folder.glob("*.jpg"))) >= target:
                    break
                try:
                    img = Image.open(f).convert("RGB")
                    
                    # Variation 1: Horizontal Flip
                    v1 = ImageOps.mirror(img)
                    v1.save(folder / f"aug_flip_{added:03d}.jpg", "JPEG", quality=92)
                    added += 1
                    
                    if len(list(folder.glob("*.jpg"))) >= target: break
                    
                    # Variation 2: Brightness +
                    enh = ImageEnhance.Brightness(img)
                    v2 = enh.enhance(1.08)
                    v2.save(folder / f"aug_bright_{added:03d}.jpg", "JPEG", quality=92)
                    added += 1
                    
                    if len(list(folder.glob("*.jpg"))) >= target: break
                    
                    # Variation 3: Slight Rotation +3 deg
                    v3 = img.rotate(3, resample=Image.BICUBIC, expand=False)
                    v3.save(folder / f"aug_rot_p_{added:03d}.jpg", "JPEG", quality=92)
                    added += 1
                    
                    if len(list(folder.glob("*.jpg"))) >= target: break
                    
                    # Variation 4: Slight Rotation -3 deg
                    v4 = img.rotate(-3, resample=Image.BICUBIC, expand=False)
                    v4.save(folder / f"aug_rot_m_{added:03d}.jpg", "JPEG", quality=92)
                    added += 1
                    
                    if len(list(folder.glob("*.jpg"))) >= target: break
                    
                    # Variation 5: Slight Contrast
                    c_enh = ImageEnhance.Contrast(img)
                    v5 = c_enh.enhance(1.06)
                    v5.save(folder / f"aug_cont_{added:03d}.jpg", "JPEG", quality=92)
                    added += 1
                except Exception as e:
                    pass

print("Augmentation finished. Checking counts:")
for folder in sorted(DATASET.iterdir()):
    if folder.is_dir() and not folder.name.startswith("_"):
        print(f"  {folder.name}: {len(list(folder.glob('*.jpg')))} images")
