# K3s Client Node Setup

This document describes how to add the client/worker nodes to the existing K3s cluster.

The K3s master node is:

```text
dli-control-plane
<CONTROL_PLANE_IP>
```

The K3s API endpoint is:

```text
https://<CONTROL_PLANE_IP>:6443
```

## Worker Nodes

| Physical Node | Recommended K3s Node Name | Interface | Node IP |
|---|---|---|---|
| `dli-worker-1` | `dli-worker-1` | `wlan0` | `<WORKER_1_IP>` |
| `dli-worker-2` | `dli-worker-2` | `wlan0` | `<WORKER_2_IP>` |
| `dli-worker-3` | `dli-worker-3` | `wlan0` | `<WORKER_3_IP>` |
| `dli-worker-4` | `dli-worker-4` | `wlan0` | `<WORKER_4_IP>` |

Use unique node names. Even if each machine shows the shell prompt `dli-worker`, the Kubernetes node name should be different for every worker.

## 1. Verify Network On Each Client

Run this on each worker:

```bash
ip addr show wlan0
hostname
hostname -I
```

Expected client IPs:

```text
dli-worker-1   <WORKER_1_IP>
dli-worker-2   <WORKER_2_IP>
dli-worker-3   <WORKER_3_IP>
dli-worker-4  <WORKER_4_IP>
```

Verify each worker can reach the master:

```bash
ping -c 3 <CONTROL_PLANE_IP>
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

## 5. Join `dli-worker-1`

Run on `dli-worker-1`:

```bash
sudo env K3S_URL="https://<CONTROL_PLANE_IP>:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name dli-worker-1 --node-ip <WORKER_1_IP>" \
sh /tmp/install-k3s.sh
```

## 6. Join `dli-worker-2`

Run on `dli-worker-2`:

```bash
sudo env K3S_URL="https://<CONTROL_PLANE_IP>:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name dli-worker-2 --node-ip <WORKER_2_IP>" \
sh /tmp/install-k3s.sh
```

## 7. Join `dli-worker-3`

Run on `dli-worker-3`:

```bash
sudo env K3S_URL="https://<CONTROL_PLANE_IP>:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name dli-worker-3 --node-ip <WORKER_3_IP>" \
sh /tmp/install-k3s.sh
```

## 8. Join `dli-worker-4`

Run on `dli-worker-4`:

```bash
sudo env K3S_URL="https://<CONTROL_PLANE_IP>:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name dli-worker-4 --node-ip <WORKER_4_IP>" \
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
dli-control-plane   Ready    control-plane,etcd   <CONTROL_PLANE_IP>
dli-worker-1      Ready    <none>               <WORKER_1_IP>
dli-worker-2      Ready    <none>               <WORKER_2_IP>
dli-worker-3      Ready    <none>               <WORKER_3_IP>
dli-worker-4     Ready    <none>               <WORKER_4_IP>
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

sudo env K3S_URL="https://<CONTROL_PLANE_IP>:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name NODE_NAME --node-ip NODE_IP" \
sh /tmp/install-k3s.sh
```

