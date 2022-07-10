import os
import time
import shutil
import tqdm

def still_running(subprocesses):
    for subp in subprocesses:
        if subp.poll() is None:
            return True
    return False

class ProgressTracker():
    """
    A lightweight progress tracker that requires that some worker fills up a
    temporary progress folder with empty files to signify progress.
    """
    
    def __init__(
        self,
        total_steps, 
        progress_dir,
        bar_width = 100,
        refresh_rate=0.5,
    ):
        assert "TMP" in progress_dir, "progress_dir must be a temporary folder"
    
        self.total_steps = total_steps
        self.progress_dir = progress_dir 
        self.bar_width = 80
        
        if os.path.isdir(self.progress_dir):
            shutil.rmtree(self.progress_dir)
        os.mkdir(self.progress_dir)
        
        self.refresh_rate = refresh_rate
        
        self.current_step = 0 # this will match # of files in TMP directory
        self.start_time = None
    
    def monitor(self, subprocesses):
        pbar = tqdm.tqdm(total=self.total_steps, ncols=self.bar_width)
        while still_running(subprocesses):
            pbar.n =  len(os.listdir(self.progress_dir))
            pbar.refresh()
            time.sleep(self.refresh_rate)
        pbar.n = self.total_steps
        pbar.refresh()
        shutil.rmtree(self.progress_dir)