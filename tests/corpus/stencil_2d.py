"""2D 5-point stencil corpus: output = average of 4 neighbors"""

def stencil_2d(input_arr, output, rows, cols):
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            output[i * cols + j] = (
                input_arr[(i - 1) * cols + j]
                + input_arr[(i + 1) * cols + j]
                + input_arr[i * cols + (j - 1)]
                + input_arr[i * cols + (j + 1)]
            ) / 4.0
    return output