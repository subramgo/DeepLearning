## Data files

create a folder 'data' inside the directory 'python'

* Adience dataset
	* download adience, curl -O -u adiencedb:adience http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/faces.tar.gz
	* tar xvf faces.tar.gz
	* Download labels (fold_0_data, fold_1_data,.....fold_4_data.txt)
		* curl -O -u adiencedb:adience http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_3_data.txt

	* Run img-process/adience-clean-label.py, this will create a file called 'labels.csv' in /data/Adience folder

	* run img-process/adience-100.py to create adience-100.h5 in /data/Adience/hdf5 folder
	   This has the images cropped to 100 x 100 and saved along with labels.
	   There are two lables, 1 where we have the age and gender separtately stored
	   Another label where we have a 16 dim Y variable.
