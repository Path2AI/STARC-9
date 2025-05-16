DeepCluster++
---------------
Step 1: Choose a set of representative WSIs for building a training set, extract tiles of size 256x256, preprocess them, and store them in the following folder structure. Make sure the tiles are RGB.

        Dataset
           |
	   |---WSI1--- 
		|---image1.png
		|---image2.png
		|    ....
	
	   |---WSI2--- 
		|---image1.png
		|---image2.png
		|    ....
	   |  .....

---------------------------------
Step 2: Create a virtual environment and install the necessary package.

conda create -n DeepCluster++ python=3.12 (any version after 3.10)

conda activate DeepCluster++

pip install torch torchvision tqdm numpy pandas matplotlib seaborn scikit-learn scipy pillow

---------------------------------

Step 3: Pass the WSI folder "Dataset" and values for other parameters through the command line.      

python DeepCluster++.py --input /path/Dataset --output /path/Output --batch-size 128 --samples-per-cluster 400 --distance-groups 5 --sample-percentage 0.20  

Other parameters that can be passed along with the command:
To make use of GPU: --device cuda 
To specify the GPU ID if available: --device cuda --gpu-id 1 
To specify the AE_CRC model  --model /path/to/custom/model.pth --seed 123

The Output folder contains the following folder structure.

        Output
           |---clusters
           |---features
           |---plots
           |---samples

	cluster (contains clusters of each WSI before sampling)
           |---WSI1
		|---Cluster_0
		|---Cluster_1
		|---  ....
	   |---WSI2
	   |--- ....

	features (2 csv files: features of each image and its cluster assignment)
            |---WSI1
		|---cluster_assignments.csv
		|---features.csv 
	   |---WSI2
	   |--- ....

	plots (2 image files: t-sne visualization with k-means cluster - with and without cluster number)
            |---WSI1
		|---tsne_with_legend
		|---tsne_with_numbers
	   |---WSI2
	   |--- ....

	samples (every cluster of a WSI contains a set of groups with a set of samples)
            |---WSI1
                |---Cluster_0
		      |---Group_0
		      |---Group_1
		      |---Group_2
		      |---Group_3
		      |---Group_4 
                |---Cluster_2
                |--- ....
	   |---WSI2
	   |--- .... 

---------------------------------

Step 4: Explore the groups from different clusters of each WSI for possible tile collection under different tissue types ADI, LYM, MUS, FCT, MUC, NCS, BLD, TUM, NOR.

---------------------------------

Step 5: The training set is verified by pathologists.

---------------------------------

Step 6: Perform tile normalization on the finalized dataset. We applied Macenko Normalization

---------------------------------
