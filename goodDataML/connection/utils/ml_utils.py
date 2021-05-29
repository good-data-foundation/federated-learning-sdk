

def bytes_from_file(path, chunk_size):
    with open(path, 'rb') as file_object:
        while True:
            chunk_data = file_object.read(chunk_size)
            if not chunk_data:
                break
            yield chunk_data
