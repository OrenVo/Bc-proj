import os
import cv2 as cv


class FileReader:
    def __init__(self, directory, files=None):
        """
        :type directory: str
        :type files: list
        """
        self.directory = None if directory is None else directory
        if self.directory:
            if self.directory[-1] != '/':
                self.directory += '/'
        self.files = files if files is not None else list()

    def next_file(self):
        if not self.files:
            return None
        else:
            file = self.files[0]
            self.files.pop(0)
            return cv.imread(file), file

    def read_dir(self):
        # List directory and subdirectories
        # List all files and save path + file to files atribute
        if not self.directory:
            return

        for (path, directories, files) in os.walk(self.directory):
            self.files += [os.path.join(path, f) for f in files]
