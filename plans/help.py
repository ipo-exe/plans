"""
Script to open the PLANS documentation locally.
Note: documentation is expected to live in `./plans/docs/_build`

"""
import os
import webbrowser
# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Define the path to your HTML file
html_file_path = "{}/docs/_build/index.html".format(os.path.dirname(script_dir))


def main():
    # Check if the file exists
    if os.path.exists(html_file_path):
        # Open the HTML file in a new browser window
        webbrowser.open_new(html_file_path)
        print(f"Opened '{html_file_path}' in a new browser window.")
    else:
        print(f"Error: The file '{html_file_path}' does not exist.")

if __name__ == "__main__":
    main()