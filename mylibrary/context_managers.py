#!/usr/bin/env python3
# context_manager.py

class JSONContextManager:
    """ 
    Customized context manager to handle a file such as TXT or JSON 
    Context managers ensure file is closed and resources are released
    example usage:
    with JSONFileManager(jsonFileLoc, 'r') as data_file:
        intents = json_load(data_file)
    """
    
    def __init__(self, file_name, mode):
        self.file_name = file_name
        self.mode = mode
        self.__file = None

    def __enter__(self):
        """ Open the actual file for use """
        try:
            self.__file = open(self.file_name, self.mode)
            return self.__file
        except:
            print(f'Context Manager - no file found: {self.file_name}')
            

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ ensures file is closed when the with block is completed """
        try:   
            if not self.__file.closed:
                self.__file.close()
        except:
            print(f"Context Manager - exit wasn't applied {self.file_name}")
        