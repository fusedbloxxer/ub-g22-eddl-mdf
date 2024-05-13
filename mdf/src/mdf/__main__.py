from . import *
from .data import DataManager

# Download standard shapes
data_manager = DataManager(download=True, data_dir=data_dir, cache_dir=cache_dir)

# Plot shapes
print(data_manager.objects.woody.content.vertices)