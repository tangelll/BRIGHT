from pathlib import Path


# =========================
# PATH'LER
# =========================

TRAIN_DIR = Path(r"D:\DatasetBright\dfc25_track2_trainval\train")
VAL_DIR   = Path(r"D:\DatasetBright\dfc25_track2_trainval\val")
TEST_DIR  = Path(r"D:\DatasetBright\dfc25_track2_test")


# =========================
# TXT OLUŞTURMA
# =========================

def create_file_list(source_dir):
    output_file = source_dir / "files.txt"

    files = sorted(source_dir.rglob("*.tif"))

    with open(output_file, "w", encoding="utf-8") as f:
        for file in files:

            name = file.name

            # Dosya uzantısını kaldır
            name = name.removesuffix(".tif")

            # Pre/post disaster kısmını kaldır
            name = name.removesuffix("_pre_disaster")
            name = name.removesuffix("_post_disaster")
            name = name.removesuffix("_building_damage")


            f.write(name + "\n")

    print(f"\n{source_dir}")
    print(f"Toplam dosya: {len(files)}")
    print(f"Oluşturuldu: {output_file}")


# =========================
# TRAIN / VAL / TEST
# =========================

create_file_list(TRAIN_DIR)
create_file_list(VAL_DIR)
create_file_list(TEST_DIR)

print("\nTüm files.txt dosyaları oluşturuldu.")