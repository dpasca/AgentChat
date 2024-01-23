#==================================================================
# SessionDict.py
#
# Author: Davide Pasca, 2024/01/23
# Description: `session` dictionary to simulate Flask's session
#==================================================================

from logger import *
import json

class SessionDict(dict):
    def __init__(self, filename, *args, **kwargs):
        self.filename = filename
        self.modified = False
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                super(SessionDict, self).__init__(data, *args, **kwargs)
        else:
            super(SessionDict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        self.modified = True
        super(SessionDict, self).__setitem__(key, value)
        self.saveToDisk()

    def __delitem__(self, key):
        self.modified = True
        super(SessionDict, self).__delitem__(key)
        self.saveToDisk()

    def saveToDisk(self):
        if self.modified:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with open(self.filename, 'w') as f:
                json.dump(self, f)
            self.modified = False
            logmsg(f"Saved session to {self.filename}")
