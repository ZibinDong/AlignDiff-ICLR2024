# AlignDiff

An example for walker-syn training

```bash
python train_attr_func.py --task walker --label_type syn  --device [YOUR_DEVICE]
python train_diffusion_model.py --task walker --label_type syn --device [YOUR_DEVICE]
python eval.py --task walker --label_type syn --device [YOUR_DEVICE]
python plot.py --task walker
```