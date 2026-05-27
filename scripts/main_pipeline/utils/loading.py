# loading.py
from progress.bar import ChargingBar

def create_bar(label, total):
    """
    Create a standardized progress bar for Omni2LOD3.
    """
    return ChargingBar(
        label,
        max=total,
        suffix="%(percent)d%%",
        fill="█",
        empty_fill="·"
    )
