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

A key names one machine (`runner_scripts/bench_key.py`); two machines never share a key. `trials/`, `logs/`, `plots/`, the build caches and the in-flight `*.partial` files and `*.lock` directories beside a leg are not synced.

## Remote

The Linux box is `chris-linux-dual` on the tailnet (100.106.157.54, user `cca79`); the tree is `/srv/gpuode/data`, owned by `cca79`, and sshd admits `cca79` from tailnet addresses (100.64.0.0/10) by key only. On every other machine `box` is its name both as a `Host` in `~/.ssh/config` and as the sftp rclone remote `box:`, so `box:/srv/gpuode/data` names the tree for rclone and rsync alike; `GPUODE_STORE_REMOTE` overrides it.

On the box, once, as root:

```
apt-get install -y openssh-server
printf 'PubkeyAuthentication yes\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nAllowUsers cca79@100.64.0.0/10 cca79@127.0.0.1\n' > /etc/ssh/sshd_config.d/10-gpuode.conf
mkdir -p /run/sshd && sshd -t && systemctl enable --now ssh && systemctl restart ssh
mkdir -p /srv/gpuode/data && chown -R cca79:cca79 /srv/gpuode
```

Then, as `cca79`: each other machine's public key on its own line in `~/.ssh/authorized_keys`, and `rclone` and `rsync` on `PATH` (the rclone binary under `~/.local/bin` needs no root). The first tree is the committed `data/` of a checkout: `rsync -a --exclude '*.lock' --exclude '*.partial' data/ /srv/gpuode/data/`.

On each other machine: `rclone` on `PATH` (Windows: `winget install Rclone.Rclone`), then

```
rclone config create box sftp host chris-linux-dual user cca79 key_file ~/.ssh/id_ed25519 known_hosts_file ~/.ssh/known_hosts
```

and in `~/.ssh/config` a `Host box` with `HostName chris-linux-dual`, `User cca79` and the same key; run `ssh box true` once to record the host key.

## Sync

Push your own key, then pull everything else, after every `bench.py run` and before every analysis. `bench.py run` does both after its runners (`--no-sync` skips it; a machine without rclone or the `box:` remote reports that and exits as the runners did; a failed sync exits 1). By hand:

```
python runner_scripts/sync.py push    # this key mirrored (remote files gone locally are deleted) and this key's clocks files copied
python runner_scripts/sync.py pull    # every other key and every clocks file copied in; nothing deleted
python runner_scripts/sync.py sync    # push then pull
python runner_scripts/sync.py check   # the differences under this key, exit 1 when there are any
```

`--root`, `--remote`, `--key` and `--dry-run` override the tree, the remote, the machine key and whether anything is written. It runs rclone, or rsync when rclone is absent:

```
KEY=$(python runner_scripts/bench_key.py)
rclone sync data/key=$KEY box:/srv/gpuode/data/key=$KEY --exclude "*.partial" --exclude "*.lock" --exclude "*.lock/**"
rclone copy data/clocks box:/srv/gpuode/data/clocks --include "*_$KEY.csv"
rclone copy box:/srv/gpuode/data data --exclude "key=$KEY/**" --exclude "*.partial" --exclude "*.lock" --exclude "*.lock/**"
```

```
rsync -a --delete --exclude "*.partial" --exclude "*.lock" data/key=$KEY/ box:/srv/gpuode/data/key=$KEY/
rsync -a --include "*_$KEY.csv" --exclude "*" data/clocks/ box:/srv/gpuode/data/clocks/
rsync -a --exclude "key=$KEY/" --exclude "*.partial" --exclude "*.lock" box:/srv/gpuode/data/ data/
```

Push deletes remote files gone from your key; the clocks push copies only your own files; pull excludes your key and deletes nothing; a row under another key is cleared on that machine.

## Check

```
python runner_scripts/sync.py check
python runner_scripts/store.py query "SELECT key, package, count(*) AS rows FROM results GROUP BY 1, 2 ORDER BY 1, 2"
```

The analyses read the local mirror. The committed `data/` tree is a snapshot of the remote: pull, then commit it in a data PR.
