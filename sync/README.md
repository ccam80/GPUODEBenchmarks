# The store

One `data/` tree on a store box over Tailscale, `/srv/gpuode/data` by default. Each machine keeps an untracked `data/` mirror and writes only its own `key=<os>_<gpu>` partition and its own `clocks/*_<key>.csv` files.

## Set up the box

As root on the box, with the public key files of every machine that will connect:

```
sudo bash sync/store_box.sh <user> <pubkey.pub>...
```

It installs openssh-server and rsync, allows only `<user>` by key from the tailnet and localhost, appends each key to `<user>`'s `authorized_keys`, and creates `/srv/gpuode/data` owned by `<user>`. Runs again without harm. For the box's own runs, `rclone config create box alias remote /` makes the default remote name resolve locally.

## Connect a machine

1. Join the tailnet.
2. Give the box your public key (a `store_box.sh` argument, or a line in `<user>`'s `authorized_keys`).
3. `ssh-keyscan -t rsa,ecdsa,ed25519 <box> >> ~/.ssh/known_hosts`
4. `rclone config create box sftp host <box> user <user> key_file ~/.ssh/id_ed25519 known_hosts_file ~/.ssh/known_hosts`

A `Host box` entry in `~/.ssh/config` lets rsync stand in when rclone is absent. `GPUODE_STORE_REMOTE` overrides `box:/srv/gpuode/data`.

## Sync

`bench.py run` pulls the tree before planning and pushes this key after the runners; the analyses pull before reading. Both refuse to run without the store unless `--no-sync`. By hand:

```
python sync/sync.py pull     # the whole tree into data/, nothing deleted
python sync/sync.py push     # this key and its clocks files up, nothing deleted
python sync/sync.py sync     # push, then pull
python sync/sync.py prune    # this key mirrored: files gone locally are deleted on the box; refuses an empty partition
python sync/sync.py check    # the differences under this key, exit 1 when there are any
```

`--root`, `--remote`, `--key`, `--tool rclone|rsync` and `--dry-run` override the tree, the remote, the machine key, the program and whether anything is written. In-flight `*.partial` files and `*.lock` directories are never shipped.
