# PPI Tutorials

These tutorials assume that you have a basic knowledge of Python, and that you are familiar with the libraries Numpy and Pandas and their data structures (arrays and DataFrames).
They are numbered in progressive order to take you from the data pre-processing steps, all the way to calibrating the model parameters and performing simulation experiments to detect structural bottlenecks.
More tutorials will be added on demand.
There are four ways to setup PPI.

## Direct download

1. Download the `tutorials` folder in your local machine.
2. Open the Jupyter notebooks.
3. Enjoy PPI.


## Installing via PyPI

1. Download the `tutorials` folder in your local machine.
2. In the command line, type `pip install policy_priority_inference`.
3. Open the Jupyter notebooks.
4. Remove the tutorials' section where a request is made to download PPI (this section exists only in tutorials 2 and up), and leave the command `import policy_priority_inference as ppi`.
5. Enjoy PPI.


## Cloud computing (Google Colab)

1. Create a cloud-computing session.
2. Open the tutorial of interest by providing the link to the relevant notebook in this repository.
3. You should be able to run every tutorial up to the point in which the data files are saved.
4. To avoid getting an error at this point, you need to create an empty folder named `clean_data` in the session that is running the tutorial.
Alternatively, you can just comment the line that saves the file, as all the necessary data are always copied from this repository.
5. Enjoy PPI.

## Git clone

1. Use Git to clone this repository to your local machine.
2. Run `pip install -r requirements.txt`, preferably within a virtual environment of your choice. 
3. Open the Jupyter notebooks.
4. Enjoy PPI.
