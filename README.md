# Compass
## About

This is the code for Compass: A Min-Max Invariant Detection Approach to Patch Robustness Certification for Deep Learning Models.



## Requirement

We test the code on python==3.8.16, torch==2.0.1,timm==0.6.13



## How to use

Firstly, users should download the corresponding checkpoint from https://drive.google.com/drive/folders/1Ewks-NgJHDlpeAaGInz_jZ6iczcYNDlN (Thanks for PC's good open source.) and put it into ./checkpoints or set by users.

In Compass, we can see Compass.py and pc_certification.py. pc_certification.py is used to generate the intermedia data based on the checkpoint for further operation. For example, if we want to test on ImageNet  with a 25% patch based on VIT, we shound run

```python
pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset imagenette --num_img -1 --num_mask 6 --patch_size 112
```

Then dump files are generated when finished. The dump files name in the main experiments are written in Compass.py with comments. In this example, users should set 

```python
prediction_map_list = joblib.load(os.path.join(DUMP_DIR,"prediction_map_list_two_mask_imagenet_vit_base_patch16_224_cutout2_128_m(130, 130)_s(19, 19)_50000.z"))

orig_prediction_list = joblib.load(os.path.join(DUMP_DIR,"orig_prediction_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))

label_list = joblib.load(os.path.join(DUMP_DIR,"label_list_imagenet_vit_base_patch16_224_cutout2_128_50000.z"))
````

and run the Compass.py.

## Acknowledgment

This implementation is partly based on [PC](https://github.com/inspire-group/PatchCleanser/tree/main).
