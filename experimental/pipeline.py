import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import DEFAULT_CONFIG, resolve_device
from image_captioning import load_image_captioner, generate_symptom_description


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def find_images(images_dir: Path) -> List[Path]:
    images: List[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images


def process_images_directory(
    images_dir: Path,
    output_file: Path,
    device: Optional[str] = None,
) -> pd.DataFrame:
    if device is None:
        device = resolve_device(DEFAULT_CONFIG.device)

    print(f"Device utilisé pour la pipeline IA : {device}")

    print("Chargement du modèle de captioning d'images...")
    captioner = load_image_captioner(device=device)

    images = find_images(images_dir)
    if not images:
        print(f"✗ Aucune image trouvée dans {images_dir}")
        return pd.DataFrame()

    print(f"{len(images)} image(s) trouvée(s) dans {images_dir}\n")

    rows = []
    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {image_path.name}")

        description = generate_symptom_description(
            image_path, 
            captioner=captioner,
            device=device
        )
        if not description:
            print("  ✗ Impossible de générer une description")
            description = ""
        else:
            print("  ✓ Description agronomique générée")

        rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "description": description or "",
            }
        )

    df = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n✓ Descriptions sauvegardées dans {output_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Génère des descriptions de symptômes de maladies de vignes "
            "à partir d'un dossier d'images, en utilisant une pipeline "
            "IA totalement gratuite (modèles open-source Hugging Face)."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Dossier contenant les images à analyser",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="descriptions_ia.csv",
        help="Fichier CSV de sortie (défaut: descriptions_ia.csv)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device à utiliser (auto, cpu, cuda). Défaut: auto",
    )

    args = parser.parse_args()

    images_dir = Path(args.input)
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"✗ Le dossier d'images n'existe pas ou n'est pas un dossier : {images_dir}")
        return

    DEFAULT_CONFIG.device = args.device

    output_path = Path(args.output)
    if not output_path.is_absolute():
        script_dir = Path(__file__).parent
        output_path = script_dir / output_path

    process_images_directory(images_dir=images_dir, output_file=output_path, device=args.device)


if __name__ == "__main__":
    main()

