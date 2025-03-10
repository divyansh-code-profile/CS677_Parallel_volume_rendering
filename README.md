# MPI-based Ray Casting and Visualization

The solution to this problem statment performs 3D volume rendering using ray casting. The code distributes the volume data across multiple MPI processes, applies color and opacity transfer functions, and renders the final image. It also uses early ray termination to optimize rendering.

## Features

- **MPI for Parallel Processing**: Utilizes `mpi4py` for distributing tasks across multiple processes.
- **Ray Casting**: Implements ray casting with color and opacity transfer functions using VTK.
- **Data Partitioning**: Supports both 1D (along x-axis) and 2D (along x and y axes) data decomposition for partitioning the 3D volume.
- **Early Ray Termination**: Optimizes rendering by terminating rays when opacity accumulates to 1.0.
- **Visualization**: Renders the final image using `PIL` and saves it as a PNG file.

## Requirements

To run the code, you need the following libraries:

- `mpi4py`
- `numpy`
- `vtkmodules` (VTK)
- `PIL` (Pillow)
- `matplotlib`
- `math`
- `sys`
- `os`
- `time`

Install the required libraries using:

 - `pip install mpi4py numpy pillow vtk matplotlib`

## Usage

To run the code use the following command:

 - `mpiexec -n <num_processes> python script.py <data_filename> <partition_type> <step_size> <x_min>  <x_max> <y_min> <y_max>`

<num_processes>: Number of MPI processes to use
<filename>: The name of the .raw file containing the volume data.
<partition_type>: 1 for 1D partitioning (x-direction only), 2 for 2D partitioning (x and y directions)
<step_size>: Step size for ray casting
<x_min>, <x_max>, <y_min>, <y_max>: Boundaries for trimming the volume data before rendering

## Result of the sample test cases run:

 - Generated image of four sample test cases run provided in the problem statment is also present in the "Sample_Test_Cases_Result" folder.



