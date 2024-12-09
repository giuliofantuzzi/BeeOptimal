#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import os

#--------------------------------------------------------------------------------
# Utility functions
#--------------------------------------------------------------------------------

# Utility function to get the path to the marker file 
def get_marker_path():
    """
    This function retrieves the directory of the current script,constructs the full path to the 
    "BeeMarker.png" file located in the "package_assets" directory, and checks if the file exists.

    Returns:
        str: The full path to the "BeeMarker.png" file.

    Raises:
        FileNotFoundError: If the marker file is not found at the expected path.
    """
    # Get the directory of the current script (plots.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to marker.png
    marker_path = os.path.join(base_dir, "./package_assets", "BeeMarker.png")
    if not os.path.exists(marker_path):
        raise FileNotFoundError(f"Marker file not found at: {marker_path}")
    return marker_path