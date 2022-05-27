python train.py --seed 23 --config configs/cnn_10.yaml --name seed23  --gpu 4
python train.py --seed 24 --config configs/cnn_10.yaml --name seed24  --gpu 4
python train.py --seed 25 --config configs/cnn_10.yaml --name seed25  --gpu 4
python train.py --seed 26 --config configs/cnn_10.yaml --name seed26  --gpu 4
python train.py --seed 27 --config configs/cnn_10.yaml --name seed27  --gpu 4
python train.py --seed 28 --config configs/cnn_10.yaml --name seed28  --gpu 4
python train.py --seed 29 --config configs/cnn_10.yaml --name seed29  --gpu 4
python train.py --seed 30 --config configs/cnn_10.yaml --name seed30  --gpu 4
python train.py --seed 31 --config configs/cnn_10.yaml --name seed31  --gpu 4
python train.py --seed 32 --config configs/cnn_10.yaml --name seed32  --gpu 4

# seeds for vggnets are different, because some of the seed don't converge during training
python train.py --seed 23 --config configs/vgg16.yaml --name seed23  --gpu 4
python train.py --seed 24 --config configs/vgg16.yaml --name seed24  --gpu 4
python train.py --seed 257 --config configs/vgg16.yaml --name seed257  --gpu 4
python train.py --seed 26 --config configs/vgg16.yaml --name seed26  --gpu 4
python train.py --seed 277 --config configs/vgg16.yaml --name seed277  --gpu 4
python train.py --seed 287 --config configs/vgg16.yaml --name seed287  --gpu 4
python train.py --seed 298 --config configs/vgg16.yaml --name seed298  --gpu 4
python train.py --seed 300 --config configs/vgg16.yaml --name seed300  --gpu 4
python train.py --seed 31 --config configs/vgg16.yaml --name seed31  --gpu 4
python train.py --seed 32 --config configs/vgg16.yaml --name seed32  --gpu 4

python train.py --seed 23 --config configs/resnet18.yaml --name seed23  --gpu 4
python train.py --seed 24 --config configs/resnet18.yaml --name seed24  --gpu 4
python train.py --seed 25 --config configs/resnet18.yaml --name seed25  --gpu 4
python train.py --seed 26 --config configs/resnet18.yaml --name seed26  --gpu 4
python train.py --seed 27 --config configs/resnet18.yaml --name seed27  --gpu 4
python train.py --seed 28 --config configs/resnet18.yaml --name seed28  --gpu 4
python train.py --seed 29 --config configs/resnet18.yaml --name seed29  --gpu 4
python train.py --seed 30 --config configs/resnet18.yaml --name seed30  --gpu 4
python train.py --seed 31 --config configs/resnet18.yaml --name seed31  --gpu 4
python train.py --seed 32 --config configs/resnet18.yaml --name seed32  --gpu 4


# execute the following cmd before run this shell script
# sed 's/\r//' -i sanity_check.sh