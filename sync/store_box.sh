#!/bin/bash
set -euo pipefail
# Store box setup: key-only tailnet sshd for USER, USER's authorized_keys from PUBKEY files, /srv/gpuode/data owned by USER, and the venv (pyarrow) that runs box_prune.py beside it after every push.
user=${1:?usage: sudo bash store_box.sh USER [PUBKEY...]}
shift
home=$(getent passwd "$user" | cut -d: -f6)
[ -d "$home" ] || { echo "no home for $user" >&2; exit 1; }
here=$(cd "$(dirname "$0")" && pwd)

apt-get install -y openssh-server rsync python3-venv

mkdir -p /etc/ssh/sshd_config.d
cat > /etc/ssh/sshd_config.d/10-gpuode.conf <<CONF
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
AllowUsers $user@100.64.0.0/10 $user@127.0.0.1
CONF
mkdir -p /run/sshd
sshd -t
if systemctl cat ssh.service >/dev/null 2>&1; then unit=ssh; else unit=sshd; fi
systemctl enable --now "$unit"
systemctl restart "$unit"

install -d -m 700 -o "$user" -g "$user" "$home/.ssh"
touch "$home/.ssh/authorized_keys"
chmod 600 "$home/.ssh/authorized_keys"
chown "$user:$user" "$home/.ssh/authorized_keys"
for pubkey in "$@"; do
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    grep -qxF -- "$line" "$home/.ssh/authorized_keys" || echo "$line" >> "$home/.ssh/authorized_keys"
  done < "$pubkey"
done

mkdir -p /srv/gpuode/data
chown -R "$user:$user" /srv/gpuode
chmod 755 /srv/gpuode /srv/gpuode/data

# The box-side pruner: sync.py ships box_prune.py here on every push and runs it with this venv.
[ -x /srv/gpuode/venv/bin/python3 ] || sudo -u "$user" python3 -m venv /srv/gpuode/venv
sudo -u "$user" /srv/gpuode/venv/bin/pip install --quiet --upgrade pip pyarrow
install -m 644 -o "$user" -g "$user" "$here/box_prune.py" /srv/gpuode/box_prune.py

ss -ltnp | awk 'NR==1 || $4 ~ /:22$/'
sshd -T | awk 'tolower($1) ~ /^(passwordauthentication|pubkeyauthentication|kbdinteractiveauthentication|allowusers)$/'
ls -ld /srv/gpuode/data
sudo -u "$user" /srv/gpuode/venv/bin/python3 /srv/gpuode/box_prune.py probe --root /srv/gpuode/data --key setup
