# ShapeNet preprocessing.
This repository provides code to fit an (ShapeNet) .obj file into the tetrahedral grid representation.

## Run single object

```python
python fit_single.py --shapenet /path/to/ShapeNetv2 --clid class_id --shid obj_id --out_dir /path/to/out
```

The script will run two rounds of optimization. In the second one, SDFs are fixed to -1, 1 and only the displacement vectors are optimized.
All parameters such as the grid size are loaded from `configs/shapenet.json`. 

## Run many objects (custom datasets)

Use `fit_many.py` to batch-convert many OBJ files into training-ready `sample.pth` files.

Expected input layout (default):

```text
<input_root>/<class_id>/<model_id>/**/*.obj
```

Training-compatible output layout:

```text
<output_root>/<class_id>/<model_id>/mesh_data/sample.pth
```

Example:

```python
python fit_many.py --input_root D:/organelles_obj --output_root D:/organelles_tetra --dmtet_grid 128 --iter 3000 --update_all_csv ../lib/all.csv
```

`--update_all_csv` appends missing `modelId` rows used by `lib/Tetradata.py` assertions.

Internally, `geometry/dmtet.py` does the heavy lifting. We use several loss functions. It might be necessary to change the weighting or remove some of the loss functions completely depending on the class.
