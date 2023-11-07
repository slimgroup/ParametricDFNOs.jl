using Pkg
Pkg.activate("./")
using Plots

gr() 

# Function to plot the matrix with colored columns and gaps between worker assignments
function plot_columns(matrix_size, workers; gap=0.4)
    plot()
    # Define colors for each data point (cube)
    data_colors = [:skyblue2, :skyblue2, :green, :green, :red, :red, :yellow, :yellow]
    for col in 1:matrix_size
        for j in 1:matrix_size
            # Determine the color based on the column index
            color_index = ((col-1) % length(data_colors)) + 1
            # Add a gap between the sets of columns by adjusting the x-coordinates
            x0 = (col-1) + ((col-1)÷(matrix_size÷workers))*gap
            x1 = col + ((col-1)÷(matrix_size÷workers))*gap
            plot!([x0, x1, x1, x0, x0], [j-1, j-1, j, j, j-1], 
                  fill=(0, 0.2, data_colors[color_index]), linecolor=:black, legend=false)
        end
    end
    xlim_max = matrix_size + (matrix_size÷workers + 1)*gap
    plot!(size=(600,600), xlim=(0, xlim_max), ylim=(0, matrix_size), 
          xticks=[], yticks=[], aspect_ratio=1)
end

# Function to plot the matrix with colored rows and gaps between worker assignments
function plot_rows(matrix_size, workers; gap=0.4)
    plot()
    # Define colors for each data point (cube)
    data_colors = [:skyblue2, :skyblue2, :green, :green, :red, :red, :yellow, :yellow]
    for i in 1:matrix_size
        for j in 1:matrix_size
            # Determine the color based on the row index
            color_index = ((j-1) % length(data_colors)) + 1
            # Calculate the y-coordinates with a gap
            y0 = (i-1) + ((i-1)÷(matrix_size÷workers))*gap
            y1 = i + ((i-1)÷(matrix_size÷workers))*gap
            plot!([j-1, j, j, j-1, j-1], [y0, y0, y1, y1, y0], 
                  fill=(0, 0.2, data_colors[color_index]), linecolor=:black, legend=false)
        end
    end
    ylim_max = matrix_size + (matrix_size÷workers + 1)*gap
    plot!(size=(600,600), xlim=(0, matrix_size), ylim=(0, ylim_max), 
          xticks=[], yticks=[], aspect_ratio=1)
end

# Set the matrix size and number of workers
matrix_size = 8
workers = 4
gap = 1 # Define the size of the gap between worker's columns

# Generate and save the images
columns_plot = plot_columns(matrix_size, workers, gap=gap)
savefig(columns_plot, "illustrations/distributed_fft_columns_color.png")

rows_plot = plot_rows(matrix_size, workers, gap=gap)
savefig(rows_plot, "illustrations/distributed_fft_rows_color.png")
