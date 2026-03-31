import os
import fsspec
import threading
import multiprocessing

# Global singletons for thread safety
_S3_FS = None
_CACHED_S3_FS = None
_FS_LOCK = threading.Lock()

# Shared counters/status for training UI (Instrumentation)
_S3_BYTES_PULLED = multiprocessing.Value('L', 0)
# A shared array for 16 workers, each showing a 32-char status string
_WORKER_STATUS_ARRAY = multiprocessing.Array('b', 16 * 32) 

class InstrumentedS3File:
    """Wrapper that tracks bytes read for real-time MB/s monitoring."""
    def __init__(self, f_obj):
        self._f = f_obj
    def read(self, *args, **kwargs):
        data = self._f.read(*args, **kwargs)
        if data:
            with _S3_BYTES_PULLED.get_lock():
                _S3_BYTES_PULLED.value += len(data)
        return data
    def readinto(self, b, *args, **kwargs):
        n = self._f.readinto(b, *args, **kwargs)
        if n:
            with _S3_BYTES_PULLED.get_lock():
                _S3_BYTES_PULLED.value += n
        return n
    def __getattr__(self, name): return getattr(self._f, name)
    def __enter__(self): return self
    def __exit__(self, *args): self._f.close()
    def close(self): self._f.close()

def get_s3_bytes_pulled():
    return _S3_BYTES_PULLED.value

def set_worker_status(worker_id, status_str):
    """Update the shared status string for a specific worker."""
    if worker_id >= 16: return
    # Truncate and pad to 32 chars
    status_bytes = status_str.encode('ascii', 'ignore')[:31].ljust(32, b' ')
    _WORKER_STATUS_ARRAY[worker_id*32 : (worker_id+1)*32] = status_bytes

def get_worker_statuses(num_workers):
    """Retrieve all active worker status strings."""
    statuses = []
    for i in range(num_workers):
        b = bytes(_WORKER_STATUS_ARRAY[i*32 : (i+1)*32])
        statuses.append(b.decode('ascii').strip())
    return statuses

def get_s3_fs():
    """Returns a basic S3 filesystem (no blockcache)."""
    global _S3_FS
    with _FS_LOCK:
        if _S3_FS is not None:
            return _S3_FS
            
        # Attempt to load credentials
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        endpoint = os.environ.get('AWS_S3_ENDPOINT')
        
        if not access_key:
            paths = [
                os.path.expanduser('~/.s3cfg'),
                os.path.expanduser('~/AppData/Roaming/s3cmd.ini'),
                r'C:\Users\silve\AppData\Roaming\s3cmd.ini'
            ]
            for p in paths:
                if os.path.exists(p):
                    with open(p, 'r') as f:
                        for line in f:
                            if 'access_key =' in line: access_key = line.split('=')[1].strip()
                            if 'secret_key =' in line: secret_key = line.split('=')[1].strip()
                            if 'host_base =' in line: endpoint = line.split('=')[1].strip()
                    break
        
        if not endpoint:
            endpoint = 's3-west.nrp-nautilus.io'
        if not endpoint.startswith('http'):
            endpoint = 'https://' + endpoint
            
        _S3_FS = fsspec.filesystem('s3', 
            key=access_key, 
            secret=secret_key, 
            client_kwargs={'endpoint_url': endpoint},
            default_block_size=16 * 1024 * 1024 # 16MB blocks (Extreme)
        )
        return _S3_FS

def get_cached_s3_fs():
    """Returns an S3 filesystem. Currently disabled blockcache due to Windows stability issues."""
    # Temporarily returning raw FS to troubleshoot hangs
    return get_s3_fs()
