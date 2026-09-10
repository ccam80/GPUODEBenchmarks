# The store

One `data/` tree on a store box over Tailscale, `/srv/gpuode/data` by default. A machine keeps an untracked `data/` mirror and writes only its own `key=<os>_<gpu>` partition and `clocks/*_<key>.csv` files.

## Set up the box

```
sudo bash sync/store_box.sh <user> <pubkey.pub>...
```

Installs openssh-server and rsync, admits `<user>` by key from the tailnet only, appends the keys to `authorized_keys`, creates `/srv/gpuode/data` owned by `<user>`; idempotent. For runs on the box itself: `rclone config create box alias remote /`.

## Connect a machine

1. Join the tailnet.
2. Give the box your public key: a `store_box.sh` argument, or a line in `<user>`'s `authorized_keys`.
3. `ssh-keyscan -t rsa,ecdsa,ed25519 <box> >> ~/.ssh/known_hosts`
4. `rclone config create box sftp host <box> user <user> key_file ~/.ssh/id_ed25519 known_hosts_file ~/.ssh/known_hosts`

A `Host box` entry in `~/.ssh/config` serves rsync when rclone is absent. `GPUODE_STORE_REMOTE` overrides `box:/srv/gpuode/data`.

## Sync

`bench.py run` pulls before planning and pushes this key after the runners; the analyses pull before reading; both refuse to run without the store unless `--no-sync`. By hand:

```
python sync/sync.py pull     # the whole tree into data/, nothing deleted
python sync/sync.py push     # this key and its clocks files up, nothing deleted
python sync/sync.py sync     # push, then pull
python sync/sync.py prune    # this key mirrored: files gone locally are deleted on the box; refuses an empty partition
python sync/sync.py check    # the differences under this key, exit 1 when there are any
```

`--root`, `--remote`, `--key`, `--tool rclone|rsync` and `--dry-run` override the tree, the remote, the machine key, the program and whether anything is written. `*.partial` files and `*.lock` directories are never shipped.
