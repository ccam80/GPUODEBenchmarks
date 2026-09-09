# Remote store

The store is the `data/` tree; the shared copy lives on the Linux box, reached over Tailscale. Every machine keeps a full local mirror and writes only its own key partition.

## Layout

```
data/key=<os>_<gpu>/package=<pkg>/results/<problem>__<algorithm>.parquet
data/key=<os>_<gpu>/package=<pkg>/finals/<trial_id>.parquet
data/key=<os>_<gpu>/package=<pkg>/optimize.csv
data/key=<os>_<gpu>/package=julia_cpu/controllers/<problem>.csv
data/clocks/*_<os>_<gpu>.csv
```

A key names one machine (`runner_scripts/bench_key.py`); two machines never share a key. `trials/`, `logs/`, `plots/` and the build caches are not synced.

## Remote

`box` is the Linux box's Tailscale name in `~/.ssh/config` and the sftp rclone remote `box:`; the tree is `/srv/gpuode/data`, writable by each machine's ssh key.

## Sync

Push your own key, then pull everything else, after every `bench.py run` and before every analysis.

```
KEY=$(python runner_scripts/bench_key.py)
rclone sync data/key=$KEY box:/srv/gpuode/data/key=$KEY
rclone sync data/clocks box:/srv/gpuode/data/clocks
rclone copy box:/srv/gpuode/data data --exclude "key=$KEY/**"
```

rsync form of the same three steps:

```
rsync -a --delete data/key=$KEY/ box:/srv/gpuode/data/key=$KEY/
rsync -a data/clocks/ box:/srv/gpuode/data/clocks/
rsync -a --exclude "key=$KEY/" box:/srv/gpuode/data/ data/
```

On Windows use PowerShell with `$KEY = python runner_scripts/bench_key.py`. Push deletes remote files gone from your key; pull excludes your key and deletes nothing; a row under another key is cleared on that machine.

## Check

```
python runner_scripts/store.py query "SELECT key, package, count(*) AS rows FROM results GROUP BY 1, 2 ORDER BY 1, 2"
rclone check data/key=$KEY box:/srv/gpuode/data/key=$KEY
```

The analyses read the local mirror. The committed `data/` tree is a snapshot of the remote: pull, then commit it in a data PR.
