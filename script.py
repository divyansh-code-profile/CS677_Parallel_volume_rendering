from mpi4py import MPI
import numpy as np
import sys
import os
from math import ceil, sqrt
from PIL import Image

import vtkmodules.all as vtk

import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

triplets_list = []
with open('color_TF.txt', 'r') as file:
    values = file.read().replace(',', '').split()

for i in range(0, len(values), 4):
    triplet = (values[i], values[i + 1], values[i + 2], values[i + 3])
    triplets_list.append(triplet)


def print_flush(message):
    """Print message and flush the output."""
    print(message)
    sys.stdout.flush()


def read_file_dimensions(filename):
    """Extract dimensions from filename."""
    base = os.path.basename(filename)
    parts = base.split('_')
    dims_part = parts[-2]
    dimensions = tuple(map(int, dims_part.split('x')))
    print_flush(f"Extracted dimensions from filename: {dimensions}")
    return dimensions


def read_raw_file(filename, dims):
    """Read raw file into numpy array with specified dimensions and float32 type."""
    print_flush("Reading data from file...")
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    x_size, y_size, z_size = dims
    data = data.reshape((z_size, x_size, y_size))
    data = data[:, ::-1, ::-1]
    print_flush("Data read successfully.")
    return data


def trim_data(data, x_min, x_max, y_min, y_max):
    """Trim the numpy array according to xy bounds."""
    print_flush(f"Trimming data to x: [{x_min}, {x_max}] and y: [{y_min}, {y_max}]")
    trimmed_data = data[:, x_min:x_max + 1, y_min:y_max + 1]
    #trimmed_data = trimmed_data[:, ::-1, ::-1]
    print_flush("Data trimmed successfully.")
    return trimmed_data


def partition_data(data, num_procs, partition_type):
    """Partition data based on the specified partition type."""
    z_size, x_size, y_size = data.shape
    sub_data = []

    if partition_type == 1:
        # Decompose along x direction only
        print_flush("Partitioning data along x direction...")
        chunk_size_x = ceil(x_size / num_procs)
        for i in range(num_procs):
            x_start = i * chunk_size_x
            x_end = min((i + 1) * chunk_size_x, x_size)
            sub_data.append(data[:, x_start:x_end, :])
        print_flush("Data partitioned along x direction.")

    else:
        # Decompose along both x and y directions
        print_flush("Partitioning data along both x and y directions...")
        for i in range(int(sqrt(num_procs)), 0, -1):
            if num_procs % i == 0:
                y_num = i
                x_num = num_procs // i
                break
        x_chunk_size = ceil(x_size / x_num)
        y_chunk_size = ceil(y_size / y_num)
        for i in range(x_num):
            for j in range(y_num):
                x_start = i * x_chunk_size
                x_end = min((i + 1) * x_chunk_size, x_size)
                y_start = j * y_chunk_size
                y_end = min((j + 1) * y_chunk_size, y_size)
                sub_data.append(data[:, x_start:x_end, y_start:y_end])
        print_flush("Data partitioned along both x and y directions.")

    return sub_data


def linear_interpolation(pairs_list, x):
    for i in range(len(pairs_list) - 1):
        x1, y1 = float(pairs_list[i][0]), float(pairs_list[i][1])
        x2, y2 = float(pairs_list[i + 1][0]), float(pairs_list[i + 1][1])

        if x1 <= x <= x2:
            y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            return y

    return 0.0


def ray_casting(data_chunk, step_size, opacity_tf):
    z_size, x_size, y_size = data_chunk.shape
    rays_terminated = 0
    output_image = np.zeros((x_size, y_size, 3), dtype=np.float32)  # 3 channels for RGB
    r_tf = vtk.vtkPiecewiseFunction()
    g_tf = vtk.vtkPiecewiseFunction()
    b_tf = vtk.vtkPiecewiseFunction()

    def calc_transfer_functions():
        """Transfer function to map input value to color"""
        for val, r, g, b in triplets_list:
            scalar_value = float(val)
            r_float = float(r)
            g_float = float(g)
            b_float = float(b)
            r_tf.AddPoint(scalar_value, r_float)
            g_tf.AddPoint(scalar_value, g_float)
            b_tf.AddPoint(scalar_value, b_float)

    calc_transfer_functions()

    def opacity_ttf(x, opacitytf):
        return opacitytf.GetValue(x)

    for x in range(x_size):
        for y in range(y_size):
            color_accum = np.zeros(3)
            opacity_accum = 0

            z = 0
            while z < z_size:
                if step_size == 1:
                    value = data_chunk[int(z), x, y]
                else:
                    z1 = int(z)
                    z2 = min(z1 + 1, z_size - 1)
                    alpha = z - z1
                    value = (1 - alpha) * data_chunk[z1, x, y] + alpha * data_chunk[z2, x, y]

                # Apply transfer functions
                r = r_tf.GetValue(value)
                g = g_tf.GetValue(value)
                b = b_tf.GetValue(value)
                color = r, g, b
                opacity = opacity_ttf(value, opacity_tf)  # Assuming opacity is the first channel

                # Composite color and opacity
                color = np.array(color)
                color_accum += (1 - opacity_accum) * color * opacity
                opacity_accum += opacity * (1 - opacity_accum)

                # Early termination if opacity accumulates to 1.0
                if opacity_accum >= 1.0:
                    rays_terminated += 1
                    break

                z += step_size

            output_image[x, y, 0] = color_accum[0] if color_accum[0] <= 1 else 1
            output_image[x, y, 1] = color_accum[1] if color_accum[1] <= 1 else 1
            output_image[x, y, 2] = color_accum[2] if color_accum[2] <= 1 else 1

    return output_image, rays_terminated


def process_data(sub_data_chunk, step_size, opacity_tf):
    """Process the chunk of data for ray casting with transfer functions."""
    return ray_casting(sub_data_chunk, step_size, opacity_tf)


def read_file_to_pairs(filename):
    pairs_list = []
    with open(filename, 'r') as file:
        values = file.read().replace(',', '').split()

    for i in range(0, len(values), 2):
        pair = (values[i], values[i + 1])
        pairs_list.append(pair)

    return pairs_list


def read_file_to_triplets(filename):
    triplets_list = []
    with open(filename, 'r') as file:
        values = file.read().replace(',', '').split()

    for i in range(0, len(values), 4):
        triplet = (values[i], values[i + 1], values[i + 2], values[i + 3])
        triplets_list.append(triplet)

    return triplets_list


def main():
    if rank == 0:
        start_time = time.time()
        if len(sys.argv) != 8:
            print_flush("Usage: python script.py <filename> <partition> <step_size> <x_min> <x_max> <y_min> <y_max>")
            sys.exit(1)

        filename = sys.argv[1]
        partition_type = int(sys.argv[2])
        step_size = float(sys.argv[3])
        x_min, x_max, y_min, y_max = map(int, sys.argv[4:8])

        num_rays = (y_max - y_min + 1) * (x_max - x_min + 1)
        dims = read_file_dimensions(filename)
        data = read_raw_file(filename, dims)

        # Trim the data based on x and y bounds
        trimmed_data = trim_data(data, x_min, x_max, y_min, y_max)
        num_rays = trimmed_data[0].size 

        # Partition the data
        sub_data_chunks = partition_data(trimmed_data, size, partition_type)
        chunk_shapes = [chunk.shape for chunk in sub_data_chunks]  # Get shapes for each chunk

        # Prepare counts and displacements for scatterv
        counts = [chunk.size for chunk in sub_data_chunks]
        displacements = [sum(counts[:i]) for i in range(size)]

        # Flatten all data
        flattened_data = np.concatenate([chunk.flatten() for chunk in sub_data_chunks])
    else:
        # On non-root ranks, initialize variables as None to receive the broadcast/scatter
        sub_data_chunks = None
        chunk_shapes = None
        step_size = None
        flattened_data = None
        counts = None
        displacements = None
    if rank == 0:
        step_size = float(sys.argv[3])
    step_size = comm.bcast(step_size, root=0)
    # Broadcast chunk shapes to all ranks
    chunk_shapes = comm.bcast(chunk_shapes, root=0)

    # Broadcast transfer functions to all ranks
    transfer_functions = comm.bcast(None, root=0)

    # Broadcast counts and displacements for scatterv
    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)

    # Scatter the data
    recv_count = counts[rank]
    recv_buffer = np.empty(recv_count, dtype=np.float32)
    comm.Scatterv([flattened_data, counts, displacements, MPI.FLOAT], recv_buffer, root=0)

    # Reshape the received data on each rank using the appropriate shape for the current rank
    sub_data_chunk = recv_buffer.reshape(chunk_shapes[rank])

    # Process data (ray casting)
    opacity_tf_pairs = read_file_to_pairs("opacity_TF.txt")
    opacity_tf = vtk.vtkPiecewiseFunction()
    for pair in opacity_tf_pairs:
        opacity_tf.AddPoint(float(pair[0]), float(pair[1]))

    local_image , rays_terminated = process_data(sub_data_chunk, step_size, opacity_tf)

     # Gather all local images to the root processor
    gathered_images = comm.gather(local_image, root=0)
    array = comm.gather(rays_terminated ,root=0)
    if rank == 0:
        # Combine all sub-images into the final image
        print_flush("Stitching images.")
        z_size, x_size, y_size = trimmed_data.shape
        final_image = np.zeros((trimmed_data.shape[1], trimmed_data.shape[2], 3), dtype=np.float32)
        if partition_type==2:
            for i in range(int(sqrt(size)), 0, -1):
                if size % i == 0:
                    y_num = i
                    x_num = size // i
                    break
            z_size, x_size, y_size = trimmed_data.shape
            x_size1 = ceil(x_size / x_num)
            y_size1 = ceil(y_size / y_num)
        else:
            x_num=size
            y_num=1
            x_size1 = ceil(x_size / x_num)
            y_size1 = ceil(y_size / y_num)
        part=0
        for i in range(x_num):
            for j in range(y_num):
                sub_image = gathered_images[part]
                part+=1
                x_start = i * x_size1
                y_start = j * y_size1
                final_image[x_start:x_start+sub_image.shape[0], y_start:y_start+sub_image.shape[1]] = sub_image
        
        # Display the final image using PIL
        #print(final_image)

        img = Image.fromarray((final_image * 255 / final_image.max()).astype('uint8'))
        #img.show()
        img.save('rendered_image1.png')

        print_flush("Final image displayed.")
        early_ray_termination_count = 0
        for i in range(size):
            early_ray_termination_count = early_ray_termination_count + array[i]
        #print(array)
        print("Number of early terminated rays are: ", early_ray_termination_count) 
        print("Percentage of early terminated rays are: ", (early_ray_termination_count / num_rays) *100) 
        end_time = time.time()
        total_time = end_time - start_time
        print("Total time taken: ", total_time)


if __name__ == "__main__":
    main()