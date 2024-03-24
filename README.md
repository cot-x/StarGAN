# StarGAN
Unofficial custmized StarGAN.

## python StarGAN.py --help
```
usage: StarGAN.py [-h] [--image_dir IMAGE_DIR] [--image_size IMAGE_SIZE] [--result_dir RESULT_DIR]
                  [--weight_dir WEIGHT_DIR] [--lr LR] [--mul_lr_dis MUL_LR_DIS] [--num_epochs NUM_EPOCHS]
                  [--batch_size BATCH_SIZE] [--lambda_cls LAMBDA_CLS] [--lambda_recon LAMBDA_RECON] [--cpu]
                  [--generate GENERATE] [--noresume]

options:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
  --image_size IMAGE_SIZE
  --result_dir RESULT_DIR
  --weight_dir WEIGHT_DIR
  --lr LR
  --mul_lr_dis MUL_LR_DIS
  --num_epochs NUM_EPOCHS
  --batch_size BATCH_SIZE
  --lambda_cls LAMBDA_CLS
  --lambda_recon LAMBDA_RECON
  --cpu
  --generate GENERATE
  --noresume
```

### for example
```
python StarGAN.py --image_dir "/usr/share/datasets/image_dir"
```
and
```
python StarGAN.py --image_dir "/usr/share/datasets/image_dir" --generate 10
```

**Note:**
- resume.pkl is a file that saves learning checkpoints for resume and includes models, weight data, etc.
- If a weight.pth file exists in the current directory, the network weights will be automatically read.

