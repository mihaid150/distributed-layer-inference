# K3s Client Node Setup

This document describes how to add the client/worker nodes to the existing K3s cluster.

The K3s master node is:

```text
k3s-master
192.168.201.225
```

The K3s API endpoint is:

```text
https://192.168.201.225:6443
```

## Worker Nodes

| Physical Node | Recommended K3s Node Name | Interface | Node IP |
|---|---|---|---|
| `pinode7` | `pinode7` | `wlan0` | `192.168.201.226` |
| `pinode8` | `pinode8` | `wlan0` | `192.168.201.227` |
| `pinode9` | `pinode9` | `wlan0` | `192.168.201.228` |
| `pinode10` | `pinode10` | `wlan0` | `192.168.201.229` |

Use unique node names. Even if each machine shows the shell prompt `k3s-client`, the Kubernetes node name should be different for every worker.

## 1. Verify Network On Each Client

Run this on each worker:

```bash
ip addr show wlan0
hostname
hostname -I
```

Expected client IPs:

```text
pinode7   192.168.201.226
pinode8   192.168.201.227
pinode9   192.168.201.228
pinode10  192.168.201.229
```

Verify each worker can reach the master:

```bash
ping -c 3 192.168.201.225
```

## 2. Verify Time Sync

K3s installation downloads files over HTTPS. If the node clock is wrong, TLS validation can fail.

Run on each worker:

```bash
date
timedatectl
```

Expected:

```text
System clock synchronized: yes
NTP service: active
```

If time sync is not active:

```bash
sudo apt update
sudo apt install -y systemd-timesyncd
sudo systemctl enable --now systemd-timesyncd
sudo timedatectl set-ntp true
```

Then verify again:

```bash
date
timedatectl
```

## 3. Get The Cluster Join Token

Run this on the master:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

Use the output as the worker join token.

Do not commit the real token into documentation or source control. In commands below, it is represented as:

```text
<K3S_NODE_TOKEN>
```

## 4. Download K3s Installer On Each Client

Run on every worker:

```bash
curl -fL https://get.k3s.io -o /tmp/install-k3s.sh
ls -lh /tmp/install-k3s.sh
```

If the download fails with an SSL certificate error, fix time sync first.

## 5. Join `pinode7`

Run on `pinode7`:

```bash
sudo env K3S_URL="https://192.168.201.225:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name pinode7 --node-ip 192.168.201.226" \
sh /tmp/install-k3s.sh
```

## 6. Join `pinode8`

Run on `pinode8`:

```bash
sudo env K3S_URL="https://192.168.201.225:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name pinode8 --node-ip 192.168.201.227" \
sh /tmp/install-k3s.sh
```

## 7. Join `pinode9`

Run on `pinode9`:

```bash
sudo env K3S_URL="https://192.168.201.225:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name pinode9 --node-ip 192.168.201.228" \
sh /tmp/install-k3s.sh
```

## 8. Join `pinode10`

Run on `pinode10`:

```bash
sudo env K3S_URL="https://192.168.201.225:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name pinode10 --node-ip 192.168.201.229" \
sh /tmp/install-k3s.sh
```

## 9. Verify Worker Service

Run on each worker after installation:

```bash
sudo systemctl status k3s-agent --no-pager -l
sudo journalctl -u k3s-agent -b -n 100 --no-pager
```

Expected service:

```text
k3s-agent.service
```

Explanation:

- K3s server nodes run `k3s.service`.
- K3s worker/client nodes run `k3s-agent.service`.

## 10. Verify From Master

Run on the master:

```bash
sudo k3s kubectl get nodes -o wide
```

Expected cluster view:

```text
NAME         STATUS   ROLES                INTERNAL-IP
k3s-master   Ready    control-plane,etcd   192.168.201.225
pinode7      Ready    <none>               192.168.201.226
pinode8      Ready    <none>               192.168.201.227
pinode9      Ready    <none>               192.168.201.228
pinode10     Ready    <none>               192.168.201.229
```

Also verify system pods:

```bash
sudo k3s kubectl get pods -A -o wide
```

## 11. If A Client Was Joined With The Wrong Name Or IP

On the affected worker, uninstall the agent:

```bash
sudo /usr/local/bin/k3s-agent-uninstall.sh
```

On the master, remove the old node entry if it still exists:

```bash
sudo k3s kubectl delete node OLD_NODE_NAME
```

Then reinstall the worker with the correct `--node-name` and `--node-ip`.

## 12. Optional Token Rotation

If the join token was exposed in terminal logs, screenshots, notes, or chat, rotate it after the current workers have joined.

Run on the master:

```bash
sudo k3s token rotate
sudo cat /var/lib/rancher/k3s/server/node-token
```

Use the new token for future joins.

## Quick Command Template

Use this template for any new worker:

```bash
curl -fL https://get.k3s.io -o /tmp/install-k3s.sh

sudo env K3S_URL="https://192.168.201.225:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name NODE_NAME --node-ip NODE_IP" \
sh /tmp/install-k3s.sh
```

