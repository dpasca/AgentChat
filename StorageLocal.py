#==================================================================
# StorageLocal.py
#
# Author: Davide Pasca, 2024/01/18
# Desc: Storage class for local files
#==================================================================
import os
import inspect
import urllib.parse

class StorageLocal:
    def __init__(self, local_dir, ENABLE_LOGGING=False):
        self.local_dir = local_dir
        self.ENABLE_LOGGING = ENABLE_LOGGING

        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)

    def FileExists(self, file_name):
        file_path = os.path.join(self.local_dir, file_name)
        return os.path.isfile(file_path)

    def UploadFile(self, data_io, file_name):
        self.logmsg(f"Uploading file {file_name}...")
        file_path = os.path.join(self.local_dir, file_name)

        # Create directories if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            file.write(data_io.getvalue())

    def GetFileURL(self, file_name):
        self.logmsg(f"Getting file url for {file_name}...")
        try:
            file_path = os.path.join(self.local_dir, file_name)
            return urllib.parse.urljoin('file:', urllib.parse.quote(file_path))
        except Exception as e:
            self.logerr(e)
            return None

    def logmsg(self, msg):
        if self.ENABLE_LOGGING:
            caller = inspect.currentframe().f_back.f_code.co_name
            print(f"[{caller}] {msg}")

    def logerr(self, msg):
        if self.ENABLE_LOGGING:
            caller = inspect.currentframe().f_back.f_code.co_name
            print(f"\033[91m[ERR]\033[0m[{caller}] {msg}")

