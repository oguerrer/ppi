# PPI Tutorials

These tutorials assume that you have basic knowledge of Python, and that you are familiar with the libraries Numpy and Pandas, as well as their data structures (arrays and DataFrames).
The tutorials are numbered in progressive order to take you from the data pre-processing steps, all the way to calibrating the model parameters to running various types of prospective analyses.

If you are not familiar with programming, we have created a simplified version of the model in JavaScript, which can be accessed through [PPI's homepage](http://policypriority.org).
This app provides a graphical interface and data templates to perform quick preliminary analysis.
Due to the compromise needed to achieve a high degree of user friendliness, the online app offers less precision and flexibility than the Python library, so it would not be thought of as a replacement for these tutorials, but as a dissemination tool.

Coming back to the Python PPI library, there are three ways to setup PPI.

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
