import os


COOKED_TRACE_FOLDER = './dataset1'


def load_trace(train_folder=COOKED_TRACE_FOLDER):
    heavy_size = []
    buffer_size = []
    ARE = []
    ARE_finder = {}
    with open(train_folder, 'rb') as f:
        for line in f:
            parse = line.split()
            heavy_size.append(float(parse[0]))
            buffer_size.append(float(parse[1]))
            ARE.append(parse[2])
            ARE_finder[str(parse[0] + '_'+ str(parse[1]))] = parse[2]

    return heavy_size, buffer_size, ARE
