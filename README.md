# Understanding the Evolution in Tourist Behavior Patterns through Context-Aware Spazio-Temporal *k*-Means

Source code and datasets for the paper *Understanding the Evolution in Tourist Behavior Patterns through Context-Aware Spazio-Temporal k-Means*

## Running the code

Unzip the file `data_POI_2014_to_2022.csv.zip` in the dataset folder.

```cmd
cd src
python main.py
```

In `src` there is the source code structured as follows:

- `data_classes.py` contains the implementation of the classes and of the methods.
- `utils_py.py` contains some functions that create and check the output folder structure.
- `main.py` executes the clustering methods.

After the execution, there will be a new folder named output containing the graphs and the clustering files of each encoding.

## Authors

- Alberto Belussi - alberto.belussi@univr.it
- Anna Dalla Vecchia - anna.dallavecchia@univr.it
- Mauro Gambini - mauro.gambini@univr.it
- Sara Migliorini - sara.migliorini@univr.it
- Elisa Quintarelli - elisa.quintarelli@univr.it

All the authors are with the Department of Computer Science, University of Verona, Verona, Italy.



