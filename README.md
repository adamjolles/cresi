![Alt text](/results/images/header1.png?raw=true "Header")

<p align="center">
<img src="/results/images/cresi_logo.png" width="350" alt="CRESI">
</p>

<h1 align="center">City-scale Road Extraction from Satellite Imagery</h1>

This repository provides an end-to-end pipeline to train models to detect  routable road networks over entire cities, and also provide speed limits and travel time estimates for each roadway.  We have observed success with both [SpaceNet](https://spacenet.ai) imagery and labels, as well as Google satellite imagery with [OSM](https://openstreetmap.org) labels. The repository consists of pre-processing modules, deep learning segmentation model (inspired by the winning SpaceNet 3 submission by [albu](https://github.com/SpaceNetChallenge/RoadDetector/tree/master/albu-solution)), post-proccessing modules to extract the road networks, inferred speed limits, and travel times.  Furthermore, we include modules to scale up network detection to the city-scale, rather than just looking at small image chips.  The output of CRESI is a geo-referenced [NetworkX](https://networkx.github.io) graph, with full access to the many graph-theoretic algorithms included in this package.  
For further details see:

1. Our [WACV Paper](http://openaccess.thecvf.com/content_WACV_2020/html/Van_Etten_City-Scale_Road_Extraction_from_Satellite_Imagery_v2_Road_Speeds_and_WACV_2020_paper.html)
2. Blogs:
	1. [Large Road Networks](https://medium.com/the-downlinq/extracting-road-networks-at-scale-with-spacenet-b63d995be52d)
	2. [Road Speeds](https://medium.com/the-downlinq/inferring-route-travel-times-with-spacenet-7f55e1afdd6d)
	3. [OSM+Google Imagery](https://medium.com/the-downlinq/computer-vision-with-openstreetmap-and-spacenet-a-comparison-cc70353d0ace)
	4. [SpaceNet 5 Baseline Part 1 - Data Prep](https://medium.com/the-downlinq/the-spacenet-5-baseline-part-1-imagery-and-label-preparation-598af46d485e)
	5. [SpaceNet 5 Baseline Part 2 - Segmentation](https://medium.com/the-downlinq/the-spacenet-5-baseline-part-2-training-a-road-speed-segmentation-model-2bc93de564d7)
	6. [SpaceNet 5 Baseline Part 3 - Road Graph + Speed](https://medium.com/the-downlinq/the-spacenet-5-baseline-part-3-extracting-road-speed-vectors-from-satellite-imagery-5d07cd5e1d21)
	7. [SpaceNet 5 Speed / Performance Comparision](https://medium.com/the-downlinq/spacenet-5-winning-model-release-end-of-the-road-fd02e00b826c)

____
### Install ###

1. Download this repository

2. Build docker image (choose GPU or CPU)

                # GPU
                docker build -t cresi_gpu docker/gpu
                # CPU
                docker build -t cresi_cpu docker/cpu
	
3. Create docker container (all commands should be run in this container)

                # GPU
                nvidia-docker run -it --rm --ipc=host --name cresi_gpu cresi_gpu
                # CPU
                docker run -it --rm --ipc=host --name cresi_cpu cresi_cpu

Alternatively, you can run CRESI outside of Docker by installing the Python
dependencies directly:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
	

____
### Prep ###

1. Prepare train/test data. The `speed_masks.py` script converts imagery and
   GeoJSON labels into training masks. Replace the example paths below with the
   locations of your data.

                python cresi/data_prep/speed_masks.py \
                    --geojson_dir /path/to/geojsons \
                    --image_dir /path/to/images \
                    --output_conversion_csv_contin /path/to/conversion.csv \
                    --output_mask_dir_contin /path/to/masks

2. Edit the `.json` configuration file to select desired variables and point to
   the appropriate directories.

3. Unzip the provided pre-trained weights so the configuration files can locate
   them:

                unzip results/weights/aws_weights/fold0_best.pth.zip -d results/aws_weights/weights

   After extraction the weights will be available at
   `results/aws_weights/weights/fold0_best.pth`.


____
### Train ###

1. All at once

		cd /path_to_cresi/cresi
	
		./train.sh configs/sn5_baseline.json


2. Run commands individually

	A. Generate folds (within docker image)

		python /path_to_cresi/cresi/00_gen_folds.py configs/sn5_baseline.json

	B. Run train script (within docker image)

		python /path_to_cresi/cresi/01_train.py configs/sn5_baseline.json --fold=0
	


____
### Test ###


1. All at once

		cd /path_to_cresi/cresi
	
		./test.sh configs/sn5_baseline.json
	

2. Run commands individually


	A. Execute inference (within docker image)

		python /path_to_cresi/cresi/02_eval.py configs/sn5_baseline.json

	B. Merge predictions (if required)

		python /path_to_cresi/cresi/03a_merge_preds.py configs/sn5_baseline.json
	
	C. Stitch together mask windows (if required)

		python /path_to_cresi/cresi/03b_stitch.py configs/sn5_baseline.json

	D. Extract mask skeletons

		python /path_to_cresi/cresi/04_skeletonize.py configs/sn5_baseline.json
	
	E. Create graph

		python /path_to_cresi/cresi/05_wkt_to_G.py configs/sn5_baseline.json

	F. Infer road travel time and speed limit

		python /path_to_cresi/cresi/06_infer_speed.py configs/sn5_baseline.json
	

Outputs will look something like the image below:

![Alt text](/results/images/vegas_speed.jpg?raw=true "Header")

## Running with GPU or CPU

Execute the model inside the Docker container that matches your hardware.

```bash
# GPU
./test.sh configs/sn5_baseline.json
# CPU
./test.sh configs/dar_tutorial_cpu.json
```

## Running on Apple Silicon

To run the pre-trained model on macOS with Apple Silicon GPUs, a simplified
Dockerfile is provided at `docker/mps/Dockerfile`. Build and run the image with:

```bash
docker build -f docker/mps/Dockerfile -t cresi_mps .
docker run -it cresi_mps
```

The container installs PyTorch with CPU support. When executed on macOS the code
will automatically use the MPS backend if available. The pre-trained weights
from `results/weights/aws_weights/fold0_best.pth.zip` are extracted during the
image build.
