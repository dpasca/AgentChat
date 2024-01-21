#==================================================================
# StorageLocal.py
#
# Author: Davide Pasca, 2024/01/18
# Desc: Storage class for local files
#==================================================================
import os
import inspect
import urllib.parse
from logger import *

class StorageLocal:
    def __init__(self, local_dir):
        self.local_dir = local_dir
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)

    def FileExists(self, file_name):
        file_path = os.path.join(self.local_dir, file_name)
        return os.path.isfile(file_path)

    def UploadFile(self, data_io, file_name):
        logmsg(f"Uploading file {file_name}...")
        file_path = os.path.join(self.local_dir, file_name)

        # Create directories if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            file.write(data_io.getvalue())

    def GetFileURL(self, file_name):
        logmsg(f"Getting file url for {file_name}...")
        try:
            file_path = os.path.join(self.local_dir, file_name)
            return urllib.parse.urljoin('file:', urllib.parse.quote(file_path))
        except Exception as e:
            logerr(e)
            return None
