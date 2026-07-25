# MARIDA benchmark results

Produced by `python scripts/train_marida.py`. MARIDA's own scene-grouped
train/test split is used, so no pixels from a test scene appear in training.

Trained on 429,412 pixels, tested on 194,863 held-out pixels.

- overall accuracy: **0.828**
- macro F1 across 15 classes: **0.480**
- Marine Debris: precision **0.160**, recall **0.929**, F1 **0.273** on 381 pixels

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Sediment-Laden Water | 0.997 | 0.969 | 0.983 | 93,037 |
| Clouds | 0.874 | 0.781 | 0.825 | 32,843 |
| Turbid Water | 0.930 | 0.859 | 0.893 | 32,226 |
| Marine Water | 0.746 | 0.455 | 0.565 | 23,443 |
| Cloud Shadows | 0.348 | 0.617 | 0.445 | 3,649 |
| Shallow Water | 0.246 | 0.466 | 0.322 | 2,506 |
| Waves | 0.104 | 0.127 | 0.114 | 1,865 |
| Wakes | 0.093 | 0.380 | 0.149 | 1,570 |
| Ship | 0.273 | 0.776 | 0.403 | 1,174 |
| Sparse Sargassum | 0.586 | 0.904 | 0.711 | 881 |
| Dense Sargassum | 0.946 | 0.920 | 0.933 | 760 |
| Foam | 0.128 | 0.532 | 0.206 | 387 |
| Marine Debris | 0.160 | 0.929 | 0.273 | 381 |
| Mixed Water | 0.014 | 0.272 | 0.027 | 92 |
| Natural Organic Material | 0.260 | 0.531 | 0.349 | 49 |

Most informative features:

- `B05` 0.2385
- `MNDWI` 0.1887
- `B01` 0.1864
- `NDWI` 0.1777
- `B02` 0.1527
- `B04` 0.1048
- `B03` 0.0821
- `FAI` 0.0750
