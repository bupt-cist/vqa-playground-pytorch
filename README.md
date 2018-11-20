Code and model for the paper "Chain of Reasoning for Visual Question Answer"

# Requirements
python 3.6
pytorch >= 0.4.0
The Others are listed in the requirements.txt

# Preprocessing
Image: Faster-rcnn features
Question: SkipThoughs

# Model
Models with their conf are put into one python file under directory: config/

# Training
The training of CoR2 
>  python train.py --cf config.CoR2

# Results
![CoR2](./figure/CoR2.png)




