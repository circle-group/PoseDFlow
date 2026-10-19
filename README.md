# PoseD-Flow
Official implementation of the paper PoseD-Flow: Versatile and Controlled Flow Matching Model of Human Pose

## Create env
```bash
conda env create -f environment.yml
conda activate poserfm
```

## Training Model
### AMASS
The processed AMASS dataset used for training the model can be downloaded from [GDrive](https://drive.google.com/file/d/1TQi_wKxJU3TTcVko-oPlWvp8L12lNR7F/view).
```bash
gdown 1TQi_wKxJU3TTcVko-oPlWvp8L12lNR7F
mkdir -p data/amass
unzip AMASS.zip -d data/amass && rm AMASS.zip
```

### Body Models
We use [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) human models in our experiments. Download and save them in `data/human_model_files`.

### Train model
```bash
python train.py
```
or, with custom args
```bash
python train.py --pose_datadir path/to/amass --use_wandb False
```

### Pretrained model
The trained model and config can be downloaded from [GDrive](https://drive.google.com/file/d/13DVC2dIEBAO2l1Wu1Zz7YbRXm3dl3Qrp/view?usp=sharing).
```bash
gdown 13DVC2dIEBAO2l1Wu1Zz7YbRXm3dl3Qrp
unzip ckpt.zip && rm ckpt.zip
```

### Sample poses
```bash
python pose_generation.py --num_steps 100 --solver midpoint
```

## Downstream tasks
### Pose completion
```bash
python pose_completion.py --occ_type left_leg
python pose_completion.py --occ_type legs
python pose_completion.py --occ_type arms --num_steps 300
python pose_completion.py --occ_type trunk
```

### Motion denoising
We use HumanEva split of AMASS and HPS dataset in our experiments. Split the datasets into 60s chunks using the script provided `experiment/create_chunks.py`, and save them in `data/[HumanEva/HPS]_60frame`.
```bash
python experiment/create_chunks.py --input-dir data/HumanEva --output-dir data/HumanEva_60frame  --seq-len 60
```
To run the experiments:
```bash
python motion_denoising.py --dataset AMASS
python motion_denoising.py --dataset AMASS --noise 0.1 --num_steps 400 --temp_weight 1e-1
python motion_denoising.py --dataset HPS --num_steps 400
python motion_denoising.py --dataset HPS --noise 0.1 --num_steps 400 --temp_weight 1e-1
```

### Human Mesh Recovery
We use the EHF dataset from SMPL-X. Download the EHF dataset from [SMPL-X](https://smpl-x.is.tue.mpg.de/) and detect the 2d keypoints using [ViTPose](https://github.com/ViTAE-Transformer/ViTPose). The expected directory structure is:
```
data/ehf/
├── EHF/
│   ├── 01_align.ply
│   ├── 01_img.jpg
│   ├── 01_img.png
│   ├── 01_scan.obj
│   └── ...
└── EHF_keypoints/
    └── vitpose_keypoints/
        └── predictions/
            ├── 01_img.json
            ├── 02_img.json
            └── ...
```
To run the experiment:
```bash
python human_mesh_recovery.py 
```
We also support initialization from [CLIFF](https://github.com/haofanwang/CLIFF). Follow the steps in their README and save the poses as `data/ehf/CLIFF/smpl_poses_cliff.npz`. To run the experiment:
```bash
python human_mesh_recovery.py --cliff
```
For convenience, we provide the keypoints and CLIFF poses used in our experiments in [GDrive](https://drive.google.com/file/d/1WLbujgAyNGHyjOWThwr-p5H8uBCQ1176/view?usp=sharing).
```bash
gdown 1WLbujgAyNGHyjOWThwr-p5H8uBCQ1176
unzip ehf_data.zip && rm ehf_data.zip
```

## Citation
If you find this code helpful for your research, please cite our paper:
```BibTeX
    @inproceedings{nadar2026posedflow,
        title={PoseD-Flow: Versatile and Guided Flow Matching Model of Human Pose},
        author={Jebastin Nadar and Simone Foti and Tolga Birdal},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={21165--21175},
        year={2026}
    }
```
