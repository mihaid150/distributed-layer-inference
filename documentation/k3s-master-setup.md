# K3s Master Setup Notes

This document records the K3s master installation flow used on `k3s-master`, including the failed attempts, the diagnostics, the fixes, and the final working commands.

The target machine was an Ubuntu 24.04 ARM64 node with wireless IP:

```text
192.168.201.225
```

## 1. Initial Install Attempt

The first command attempted to install K3s as a server node:

```bash
curl -sfL https://get.k3s.io | \
INSTALL_K3S_EXEC="server \
--node-name k3s-master \
--write-kubeconfig-mode 644 \
--cluster-init \
--disable traefik \
--disable servicelb \
--flannel-backend vxlan \
--node-ip 192.168.1.10 \
--advertise-address 192.168.1.10" \
sh -
```

This did not create a systemd service:

```bash
systemctl status k3s
sudo systemctl status k3s
```

Result:

```text
Unit k3s.service could not be found.
```

Explanation:

- The install script did not complete.
- `curl -s` hid the real error.
- The requested node IP, `192.168.1.10`, was also not the actual IP of the machine.

## 2. Retry Installer With Sudo

The install was retried with root privileges:

```bash
curl -sfL https://get.k3s.io | sudo env INSTALL_K3S_EXEC="server \
--node-name k3s-master \
--write-kubeconfig-mode 644 \
--cluster-init \
--disable traefik \
--disable servicelb \
--flannel-backend vxlan \
--node-ip 192.168.1.10 \
--advertise-address 192.168.1.10" sh -
```

The service still did not exist:

```bash
sudo systemctl status k3s
sudo journalctl -u k3s -f
```

Result:

```text
Unit k3s.service could not be found.
```

Explanation:

- Running with `sudo` was necessary, but the installer script was still not being downloaded successfully.
- Because `curl -s` was used, the TLS error was hidden.

## 3. Download Installer Separately

The next step was to download the script into a file and run it with shell tracing:

```bash
curl -sfL https://get.k3s.io -o install-k3s.sh
sudo env INSTALL_K3S_EXEC="server --node-name k3s-master --write-kubeconfig-mode 644 --cluster-init --disable traefik --disable servicelb --flannel-backend vxlan --node-ip 192.168.1.10 --advertise-address 192.168.1.10" sh -x install-k3s.sh
```

Result:

```text
sh: 0: cannot open install-k3s.sh: No such file
```

Explanation:

- `install-k3s.sh` did not exist because the download failed.
- The command needed to be rerun without silent mode to expose the real problem.

## 4. Expose Curl Error

The installer was downloaded without `-s`:

```bash
curl -fL https://get.k3s.io -o /tmp/install-k3s.sh
echo $?
ls -lh /tmp/install-k3s.sh
head -5 /tmp/install-k3s.sh
```

Result:

```text
curl: (60) SSL certificate problem: certificate is not yet valid
```

Additional diagnostics:

```bash
curl -vI https://get.k3s.io
date
getent hosts get.k3s.io
```

The system clock was wrong:

```text
Wed Mar 18 20:13:39 EET 2026
```

Explanation:

- The real date was May 5, 2026.
- The node clock was behind.
- TLS certificate validation failed because the certificate appeared to be from the future.

## 5. Fix System Time

Install and enable `systemd-timesyncd`:

```bash
sudo apt update
sudo apt install -y systemd-timesyncd
sudo systemctl enable --now systemd-timesyncd
sudo timedatectl set-ntp true
```

This replaced `chrony` with `systemd-timesyncd`.

Verify the clock:

```bash
date
timedatectl
```

Expected result:

```text
Local time: Tue 2026-05-05 ... EEST
System clock synchronized: yes
NTP service: active
Time zone: Europe/Bucharest
```

Explanation:

- K3s downloads over HTTPS.
- HTTPS requires a correct local clock.
- Fixing NTP fixed certificate validation.

## 6. Download Installer Successfully

After fixing time sync, download the installer again:

```bash
curl -fL https://get.k3s.io -o /tmp/install-k3s.sh
ls -lh /tmp/install-k3s.sh
```

Expected result:

```text
/tmp/install-k3s.sh
```

Explanation:

- The installer now downloads because TLS validation succeeds.

## 7. First K3s Install With Wrong IP

The installer was then run with the original IP:

```bash
sudo env INSTALL_K3S_EXEC="server --node-name k3s-master --write-kubeconfig-mode 644 --cluster-init --disable traefik --disable servicelb --flannel-backend vxlan --node-ip 192.168.1.10 --advertise-address 192.168.1.10" sh /tmp/install-k3s.sh
```

K3s installed and created the service, but startup failed:

```text
Job for k3s.service failed because the control process exited with error code.
```

Check the service:

```bash
sudo systemctl status k3s.service --no-pager -l
sudo journalctl -u k3s.service -b -n 200 --no-pager
```

Important error:

```text
listen tcp 192.168.1.10:2380: bind: cannot assign requested address
```

Explanation:

- K3s embedded etcd tried to bind to `192.168.1.10`.
- That IP was not assigned to the host.
- Linux cannot bind a service to an address that does not exist on any local interface.

## 8. Find Actual Node IP

Check the wireless interface:

```bash
ip addr show wlan0
```

The actual IP was:

```text
inet 192.168.201.225/24
```

Explanation:

- The correct K3s `--node-ip` and `--advertise-address` must match the reachable IP of the node.
- In this setup, the master node is reached at `192.168.201.225`.

## 9. Reinstall K3s With Correct IP

If K3s is in a failed restart loop, stop it:

```bash
sudo systemctl stop k3s
```

For a fresh failed install, remove the broken K3s state:

```bash
sudo /usr/local/bin/k3s-uninstall.sh
```

Then reinstall with the real IP:

```bash
sudo env INSTALL_K3S_EXEC="server --node-name k3s-master --write-kubeconfig-mode 644 --cluster-init --disable traefik --disable servicelb --flannel-backend vxlan --node-ip 192.168.201.225 --advertise-address 192.168.201.225" sh /tmp/install-k3s.sh
```

Explanation:

- `--cluster-init` initializes the first embedded-etcd server.
- `--disable traefik` disables the bundled Traefik ingress controller.
- `--disable servicelb` disables the bundled K3s service load balancer.
- `--flannel-backend vxlan` uses VXLAN for pod networking.
- `--node-ip 192.168.201.225` sets the node's internal IP.
- `--advertise-address 192.168.201.225` sets the address advertised to other nodes and clients.

## 10. Verify Master Node

Check the service:

```bash
sudo systemctl status k3s --no-pager -l
```

Check Kubernetes nodes:

```bash
sudo k3s kubectl get nodes -o wide
```

Expected result:

```text
NAME         STATUS   ROLES                VERSION        INTERNAL-IP
k3s-master   Ready    control-plane,etcd   v1.35.4+k3s1   192.168.201.225
```

Explanation:

- `Ready` means the node successfully joined its own cluster.
- `control-plane,etcd` means this node is both a Kubernetes control-plane node and the embedded etcd datastore node.

During startup this warning appeared:

```text
loading OpenAPI spec for "v1beta1.metrics.k8s.io" failed ... 503
```

This was transient and did not block the cluster from becoming ready.

## 11. Get Worker Join Token

On the master, read the node token:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

Do not commit or publish the token. Use it as a secret when joining workers.

For documentation, represent it as:

```text
<K3S_NODE_TOKEN>
```

## 12. Join Worker Nodes

On each worker node, download the installer:

```bash
curl -fL https://get.k3s.io -o /tmp/install-k3s.sh
```

Then join the worker:

```bash
sudo env K3S_URL="https://192.168.201.225:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name WORKER_NAME --node-ip WORKER_IP" \
sh /tmp/install-k3s.sh
```

Replace:

```text
WORKER_NAME
WORKER_IP
<K3S_NODE_TOKEN>
```

Explanation:

- `K3S_URL` points workers to the master API server.
- `K3S_TOKEN` authenticates the worker to the cluster.
- `agent` installs a worker node instead of another server node.
- `--node-name` should be unique for each worker.
- `--node-ip` should be the worker's real reachable IP.

## 13. Verify Workers From Master

After joining each worker, run on the master:

```bash
sudo k3s kubectl get nodes -o wide
```

Expected result:

```text
k3s-master   Ready   control-plane,etcd   ...
worker-1     Ready   <none>               ...
worker-2     Ready   <none>               ...
```

## 14. Optional Token Rotation

If the token was exposed in terminal logs, chat, screenshots, or shared notes, rotate it after all current nodes have joined:

```bash
sudo k3s token rotate
sudo cat /var/lib/rancher/k3s/server/node-token
```

Use the new token for future worker joins.

## Final Working Master Install Command

This is the final command that worked for the master node:

```bash
sudo env INSTALL_K3S_EXEC="server --node-name k3s-master --write-kubeconfig-mode 644 --cluster-init --disable traefik --disable servicelb --flannel-backend vxlan --node-ip 192.168.201.225 --advertise-address 192.168.201.225" sh /tmp/install-k3s.sh
```

## Quick Validation Commands

Use these after installation:

```bash
sudo systemctl status k3s --no-pager -l
sudo journalctl -u k3s.service -b -n 200 --no-pager
sudo k3s kubectl get nodes -o wide
sudo k3s kubectl get pods -A
```

