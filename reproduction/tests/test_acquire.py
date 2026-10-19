from pathlib import Path

import yaml

from retina_fl.acquire import locate_odir, write_runtime_config


def test_locate_and_write_runtime_config(tmp_path: Path):
    root = tmp_path / "download"
    images = root / "ODIR-5K" / "Training Images"
    images.mkdir(parents=True)
    for index in range(1000):
        (images / f"{index}_left.jpg").touch()
    annotations = root / "ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
    annotations.touch()

    location = locate_odir(root)
    assert location["image_count"] == 1000
    assert Path(location["annotations"]) == annotations
    assert Path(location["image_dir"]) == images

    template = tmp_path / "template.yaml"
    template.write_text(
        "project:\n  seed: 1\ndata:\n  annotations: old\n  image_dir: old\n  output_dir: old\n",
        encoding="utf-8",
    )
    output = write_runtime_config(template, location, tmp_path / "runtime.yaml", tmp_path / "runs")
    resolved = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert Path(resolved["data"]["annotations"]) == annotations
    assert Path(resolved["data"]["image_dir"]) == images
