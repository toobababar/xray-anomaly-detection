# Data

This project uses the **Chest X-Ray Dataset for Respiratory Disease Classification**
from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FD8BEZ).

The dataset is not included in this repository. Download it before training.

## Download

Run the provided download script from the project root:
```bash
python scripts/download_data.py
```

This will automatically download and save the dataset to `data/5194114.npz`.

## Manual Download

If you prefer to download manually:
```bash
wget https://dataverse.harvard.edu/api/access/datafile/5194114 -O data/5194114.npz
```

## Dataset Details

- **Format:** NPZ (NumPy archive)
- **Contents:** 4D numpy array of chest X-ray images with labels and image names
- **Classes:** covid, lung_opacity, normal, viral_pneumonia, tuberculosis
- **Source:** Harvard Dataverse