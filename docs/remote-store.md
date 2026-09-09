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

| item | value |
|---|---|
| host | the Linux box's Tailscale name, as `box` in `~/.ssh/config` and the rclone remote `box:` (sftp) |
| path | `/srv/gpuode/data` |
| access | the machine's ssh key, read and write on `/srv/gpuode/data` |

## Sync

Push your own key, then pull everything else. Run the pair after every `bench.py run` and before every analysis.

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

On Windows run the same under PowerShell with `$KEY = python runner_scripts/bench_key.py` and forward slashes in the local paths.

- Push uses `sync`: a leg file or finals file deleted locally under your key disappears from the remote.
- Pull uses `copy` with your key excluded: another machine's rows never overwrite yours and nothing local is deleted.
- Clearing a row on another machine's key is done on that machine.

## Check

```
python runner_scripts/store.py query "SELECT key, package, count(*) AS rows FROM results GROUP BY 1, 2 ORDER BY 1, 2"
rclone check data/key=$KEY box:/srv/gpuode/data/key=$KEY
```

The analyses read the local mirror; pull before running them.

## Git

The committed `data/` tree is a snapshot of the remote, refreshed by a data PR after a run: pull, then commit the tree.
