"""CLI for ego2robot."""
import click
import yaml
import sys
import os

@click.group()
def cli():
    """Ego2Robot: Convert egocentric video to robot datasets."""
    pass

@cli.command()
@click.option('--config', default='config/default.yaml', help='Config file path')
@click.option('--max-videos', default=None, type=int, help='Max videos to process')
def convert(config, max_videos):
    """Convert egocentric video to LeRobot format."""
    from ego2robot.data.sampler import EgocentricSampler
    from ego2robot.data.clips import ClipExtractor
    from ego2robot.data.quality import QualityFilter
    from ego2robot.data.storage import ClipStorage
    
    # Load config
    with open(config) as f:
        cfg = yaml.safe_load(f)
    
    if max_videos:
        cfg['data']['max_videos'] = max_videos
    
    click.echo(f"Processing {cfg['data']['max_videos']} videos...")
    
    # Run pipeline
    sampler = EgocentricSampler(cfg)
    extractor = ClipExtractor(cfg)
    quality_filter = QualityFilter(cfg)
    storage = ClipStorage(cfg)
    
    all_clips = []
    for i, video in enumerate(sampler.filter_videos()):
        click.echo(f"Processing video {i+1}...")
        clips = extractor.extract_clips(video['video_bytes'], video['metadata'])
        all_clips.extend(clips)
    
    click.echo(f"Filtering {len(all_clips)} clips...")
    filtered = quality_filter.filter_clips(all_clips)
    
    click.echo(f"Saving {len(filtered)} clips...")
    storage.save_clips(filtered)
    
    click.echo("✓ Done!")

@cli.command()
@click.argument('dataset_path')
def validate(dataset_path):
    """Validate a LeRobot dataset."""
    import json
    from pathlib import Path
    
    path = Path(dataset_path)
    info_file = path / "meta" / "info.json"
    
    if not info_file.exists():
        click.echo("❌ No info.json found", err=True)
        sys.exit(1)
    
    with open(info_file) as f:
        info = json.load(f)
    
    click.echo(f"✓ Valid LeRobot dataset")
    click.echo(f"  Episodes: {info['total_episodes']}")
    click.echo(f"  Frames: {info['total_frames']}")
    click.echo(f"  FPS: {info['fps']}")

if __name__ == '__main__':
    cli()