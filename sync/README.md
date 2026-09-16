# The store

One `data/` tree on a store box over Tailscale, `/srv/gpuode/data` by default. A machine keeps an untracked `data/` mirror and writes only its own `key=<os>_<gpu>` partition and its clocks files, `clocks/calibration_<key>.csv` and the 25 Hz `clocks/<key>_<stamp>.csv` log of each run.

## Set up the box

```
sudo bash sync/store_box.sh <user> <pubkey.pub>...
```

Installs openssh-server, rsync and python3-venv, admits `<user>` by key from the tailnet only, appends the keys to `authorized_keys`, creates `/srv/gpuode/data` owned by `<user>`, and beside it `/srv/gpuode/venv` (pyarrow) and `/srv/gpuode/box_prune.py`, the box-side pruner every push runs; idempotent. For runs on the box itself: `rclone config create box alias remote /`, and the pruner then runs in-process.

## Connect a machine

1. Join the tailnet.
2. Give the box your public key: a `store_box.sh` argument, or a line in `<user>`'s `authorized_keys`.
3. `ssh-keyscan -t rsa,ecdsa,ed25519 <box> >> ~/.ssh/known_hosts`
4. `rclone config create box sftp host <box> user <user> key_file ~/.ssh/id_ed25519 known_hosts_file ~/.ssh/known_hosts`

A `Host box` entry in `~/.ssh/config` serves rsync when rclone is absent. `GPUODE_STORE_REMOTE` overrides `box:/srv/gpuode/data`.

## Sync

`bench.py run` pulls before planning (refusing when the box cannot run its pruner) and pushes this key after the runners; the analyses pull before reading; both refuse to run without the store unless `--no-sync`. A pull never replaces a local file newer than the box's; a run refuses to start while this key's local partition holds files the box lacks or differs from (`unpushed` below), until they are pushed or the partition deleted.

A push copies this key and this machine's clocks files up, deleting nothing. It ships `sync/box_prune.py` to the box, holds the key's lock there (`box_prune.py serve` over ssh, from the sftp remote's host, user and key) through the upload, then has the box delete the `clocks/<key>_<stamp>.csv` logs a day or older that no result row under `key=<key>` names; the mirror and `logs/<run>/` drop the same ones. A failed upload releases the lock without pruning; a results file the box cannot read stops the prune. By hand:

```
python sync/sync.py pull     # the whole tree into data/, nothing deleted, newer local files kept
python sync/sync.py push     # this key and its clocks files up, nothing deleted; then the box prunes this key's orphan logs
python sync/sync.py sync     # push, then pull
python sync/sync.py prune    # this key mirrored: files gone locally are deleted on the box; then the box prunes; refuses an empty partition
python sync/sync.py check    # the differences under this key, exit 1 when there are any
python sync/sync.py unpushed # this key's local files missing from or differing on the box, exit 1 when there are any
```

`--root`, `--remote`, `--key`, `--tool rclone|rsync` and `--dry-run` override the tree, the remote, the machine key, the program and whether anything is written (a dry run lists what the box would prune). `*.partial` files, `*.lock` directories and the box's `.sync_<key>.lock` files are never shipped.
