IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
      Args:
          filename (string): path to a file
      Returns:
          bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def simple_table(rows):
    lengths = [
        max(len(row[i]) for row in rows) + 1 for i in range(len(rows[0]))
    ]
    row_format = ' '.join(('{:<%s}' % length) for length in lengths[:-1])
    row_format += ' {}'  # The last column can maintain its length.

    output = ''
    for i, row in enumerate(rows):
        if i > 0:
            output += '\n'
        output += row_format.format(*row)
    return output
