import matplotlib.pyplot as plt
import os

def create_bar_chart(year_data, job_losses_data, output_filename='output/graph.png'):
    """
    Generates a bar chart showing year vs. number of job losses and saves it to a file.

    Args:
        year_data: A list or tuple of years (integers or strings).
        job_losses_data: A list or tuple of job losses corresponding to the years.  Must be same length as year_data.
        output_filename: The path to save the generated image. Defaults to 'output/graph.png'.

    Raises:
        ValueError: If input data is invalid or inconsistent.
        IOError: If there's an issue saving the file.

    """
    if not isinstance(year_data, (list, tuple)) or not isinstance(job_losses_data, (list, tuple)):
        raise ValueError("Year and job losses data must be lists or tuples.")
    if len(year_data) != len(job_losses_data):
        raise ValueError("Year and job losses data must have the same length.")
    if not all(isinstance(year, (int, str)) for year in year_data):
        raise ValueError("Years must be integers or strings.")
    if not all(isinstance(losses, (int, float)) for losses in job_losses_data):
        raise ValueError("Job losses must be numbers.")
    if any(losses < 0 for losses in job_losses_data):
        raise ValueError("Job losses cannot be negative.")


    # Create the output directory if it doesn't exist.  Robust error handling.
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            raise IOError(f"Error creating output directory: {e}") from e


    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.bar(year_data, job_losses_data, color='skyblue')
    plt.xlabel("Year")
    plt.ylabel("Number of Job Losses")
    plt.title("Job Losses Over the Years")
    plt.xticks(rotation=45, ha="right") # Rotate x-axis labels for readability if needed.
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    try:
        plt.savefig(output_filename)
        print(f"Bar chart saved to {output_filename}")
    except Exception as e:
        raise IOError(f"Error saving bar chart to file: {e}") from e


# Example Usage
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
job_losses = [1000, 1200, 800, 1500, 900, 2000, 1800, 1100]

create_bar_chart(years, job_losses)

#Example of error handling
try:
    create_bar_chart([2015,2016], [1000,1200, 1500])
except ValueError as e:
    print(f"Caught expected ValueError: {e}")

try:
    create_bar_chart([2015,2016], [-1000,1200])
except ValueError as e:
    print(f"Caught expected ValueError: {e}")
