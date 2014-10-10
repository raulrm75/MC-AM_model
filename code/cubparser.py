from cells import CubicalCell

""""cubparser

This module enable reading cubical complexes from text files using chomp format.
"""
def interval_from_cube_vertex(vertex):
    return [(v, v + 1) for v in vertex]

def intervals_iter(input_file):
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if not (line.startswith(';') or line.startswith('dimension')):
                yield interval_from_cube_vertex(
                    map(int, line.strip()[1:-1].split(',')))

def cubes_iter(input_file):
    for interval in intervals_iter(input_file):
        yield CubicalCell(interval)
        
        
if __name__ == '__main__':
    cubes = cubes_iter('tests/kleinbot.cub')
    
    
        
