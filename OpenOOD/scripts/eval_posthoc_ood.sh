
# ImageNet-1K
python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor msp \
    --save-score \
    --save-csv

python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor mls \
    --save-score \
    --save-csv

python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor react \
    --save-score \
    --save-csv

python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor residual \
    --save-score \
    --save-csv

python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor odin \
    --save-score \
    --save-csv

python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor ebo \
    --save-score \
    --save-csv

python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor vim \
    --save-score \
    --save-csv

python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor knn \
    --save-score \
    --save-csv

python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor ash \
    --save-score \
    --save-csv


python scripts/eval_ood_imagenet.py \
   --tvs-pretrained \
   --arch resnet50 \
   --postprocessor gradnorm \
   --save-score \
   --save-csv

# CIFAR-10

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor msp \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor mls \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor react \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor residual \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor odin \
   --save-score --save-csv

python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor vim \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor knn \
   --save-score --save-csv

python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ash \
    --save-score --save-csv

python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor gradnorm \
    --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_cider_net_cider_e100_lr0.5_protom0.5_default \
   --postprocessor cider \
   --save-score --save-csv

# CIFAR-100

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor msp \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor mls \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor react \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor residual \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor odin \
   --save-score --save-csv

python scripts/eval_ood.py \
    --id-data cifar100 \
    --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor vim \
   --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor knn \
   --save-score --save-csv

python scripts/eval_ood.py \
    --id-data cifar100 \
    --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ash \
    --save-score --save-csv

python scripts/eval_ood.py \
    --id-data cifar100 \
    --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor gradnorm \
    --save-score --save-csv

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_cider_net_cider_e100_lr0.5_protom0.5_default \
   --postprocessor cider \
   --save-score --save-csv