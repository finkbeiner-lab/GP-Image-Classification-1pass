

conda activate abnn
module load cuda/12.4
cd /gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/codes/ABNN-WSI-Classification/code


python3 ABNN_WSI.py --mode TENSOR --ext svs --model_pretrained True --model_type RESNET18


python3 ABNN_WSI.py --mode TRAIN --ext pth --num_epoch 50 --batch_size 2

python3 ABNN_WSI.py --mode TEST --ext pth