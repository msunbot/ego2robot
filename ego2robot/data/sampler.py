# ego2robot/data/sampler.py
from huggingface_hub import hf_hub_download
import tarfile
import json
import os

class EgocentricSampler:
    def __init__(self, config):
        self.config = config
        
    def filter_videos(self):
        """Download from multiple workers."""
        repo_id = self.config['data']['dataset_name']
        max_videos = self.config['data'].get('max_videos', 3)
        cache_dir = "./data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        count = 0
        factory = "factory_001"
        
        # Process different workers - FIXED NAMING
        workers = ["worker_001", "worker_002", "worker_003"]
        
        for worker in workers:
            if count >= max_videos:
                break
                
            try:
                # Format: factory001_worker001_part00.tar (NO underscores before numbers!)
                worker_num = worker.split('_')[1]  # "001"
                filename = f"{factory}/workers/{worker}/factory001_worker{worker_num}_part00.tar"
                
                print(f"Downloading {filename}...")
                tar_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    cache_dir=cache_dir
                )
                
                print(f"Extracting videos from {worker}...")
                
                # Extract videos from tar
                with tarfile.open(tar_path, 'r') as tar:
                    samples = {}
                    
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                            
                        name_parts = member.name.split('/')[-1].split('.')
                        if len(name_parts) < 2:
                            continue
                            
                        basename = '.'.join(name_parts[:-1])
                        ext = name_parts[-1]
                        
                        if basename not in samples:
                            samples[basename] = {}
                        samples[basename][ext] = member
                    
                    for basename, files in samples.items():
                        if 'mp4' in files and 'json' in files:
                            mp4_member = files['mp4']
                            video_file = tar.extractfile(mp4_member)
                            video_bytes = video_file.read()
                            
                            json_member = files['json']
                            json_file = tar.extractfile(json_member)
                            metadata = json.load(json_file)
                            
                            yield {
                                'video_bytes': video_bytes,
                                'metadata': metadata,
                                'sample_id': count
                            }
                            
                            count += 1
                            if count >= max_videos:
                                return
                            
            except Exception as e:
                print(f"Skipping worker {worker}: {e}")
                continue
        
        print(f"✓ Sampled {count} videos")