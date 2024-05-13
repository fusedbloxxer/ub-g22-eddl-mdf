import pathlib as pb

# Environment setup
root_dir: pb.Path = pb.Path('.')
data_dir: pb.Path = root_dir / '..' / '..' / 'data'
cache_dir: pb.Path = root_dir / '..' / '..' / 'cache'