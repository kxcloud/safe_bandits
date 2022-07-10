import os
import time
import shutil

def still_running(subprocesses):
    for subp in subprocesses:
        if subp.poll() is None:
            return True
    return False

class ProgressBar():
    """
    A lightweight progress bar that:
        (1) doesn't assume the console can flush text,
        (2) requires that some worker fills up a temporary progress folder
        with empty files to signify progress.
    """
    
    def __init__(
        self,
        total_steps, 
        progress_dir,
        title="",
        char="#", 
        refresh_rate=0.5,
        bar_length=80
    ):
        assert "TMP" in progress_dir, "progress_dir must be a temporary folder"
    
        self.total_steps = total_steps
        self.progress_dir = progress_dir 
        
        if os.path.isdir(self.progress_dir):
            shutil.rmtree(self.progress_dir)
        os.mkdir(self.progress_dir)
        
        self.title = title
        self.char = char
        self.refresh_rate = refresh_rate
        self.bar_length = bar_length
        
        self.progress = 0
    
    def monitor(self, subprocesses):
        print(self.title+"_"*(self.bar_length-len(self.title)))
        n = 0
        while still_running(subprocesses):
            n = len(os.listdir(self.progress_dir))
            if (self.bar_length * n / self.total_steps) > self.progress:
                self.progress += 1
                print("#", end="")
            time.sleep(self.refresh_rate)
        print("#"*(self.bar_length-self.progress))    
        shutil.rmtree(self.progress_dir)
            
            
        