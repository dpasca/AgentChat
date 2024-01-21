import inspect
import os

# Enable for debugging purposes
ENABLE_LOGGING = os.getenv('FORCE_ENABLE_LOGGING', '0') == '1'

#==================================================================
def logmsg(msg):
    if ENABLE_LOGGING:
        caller = inspect.currentframe().f_back.f_code.co_name
        print(f"[{caller}] {msg}")

def logerr(msg):
    if ENABLE_LOGGING:
        caller = inspect.currentframe().f_back.f_code.co_name
        print(f"\033[91m[ERR]\033[0m[{caller}] {msg}")
