import os
import random
import shutil

# routes
train_dir = "./data/train"
val_dir = "./data/val"

# validation percentage
val_split = 0.2

# classes
classes = ["Cats", "Dogs"]

for cls in classes:
    train_cls_dir = os.path.join(train_dir, cls)
    val_cls_dir = os.path.join(val_dir, cls)

    # ensure that the val folder exists
    os.makedirs(val_cls_dir, exist_ok=True)

    # list images in train/<class>
    images = os.listdir(train_cls_dir)
    total_images = len(images)

    #shuffle randomly
    random.shuffle(images)

    # calculate how many to move
    n_val = int(total_images * val_split)
    val_images = images[:n_val]

    print(f"Moviendo {n_val}/{total_images} imágenes de {cls} a validación...")

    for img in val_images:
        src = os.path.join(train_cls_dir, img)
        dst = os.path.join(val_cls_dir, img)
        shutil.move(src, dst)

print("✅ Split completed.")
