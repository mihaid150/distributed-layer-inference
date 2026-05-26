# K3s Disk Cleanup Playbook (Easy -> Hard)

This playbook is for recovering a K3s node from `DiskPressure` and getting workloads schedulable again.

Use it in order. Stop when the node is healthy.

## 0) Fast Diagnose (always run first)

```bash
kubectl describe node dli-control-plane | egrep -i "Taints|DiskPressure"
df -h
sudo du -h --max-depth=1 /var/lib/rancher/k3s | sort -h
```

If you see:
- `node.kubernetes.io/disk-pressure:NoSchedule`
- low free space on `/`

continue below.

---

## 1) Easy Cleanup (safe, no service stop)

```bash
sudo k3s crictl rmi --prune
sudo apt-get clean
sudo journalctl --vacuum-size=200M
sudo truncate -s 0 /var/lib/rancher/k3s/agent/containerd/containerd.log
df -h
```

Re-check node condition:

```bash
kubectl describe node dli-control-plane | egrep -i "Taints|DiskPressure"
```

If still under pressure, continue.

---

## 2) Reduce Churn Before Deeper Cleanup

```bash
kubectl -n inference scale deploy --all --replicas=0
kubectl -n inference delete pod --all --force --grace-period=0
```

This prevents continuous image pulls and restarts while cleaning.

---

## 3) Locate Heavy Paths

```bash
sudo du -xhd1 / | sort -h
sudo du -xhd1 /var | sort -h
sudo du -h --max-depth=2 /var/lib/rancher/k3s/agent/containerd | sort -h | tail -n 40
sudo du -h --max-depth=1 /var/lib/rancher/k3s/agent/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots | sort -h | tail -n 40
```

If `overlayfs/snapshots` is huge and GC does not reclaim, continue.

---

## 4) Hard Cleanup: Reset Agent Containerd Runtime State

> Safe for runtime cache cleanup.  
> Do **not** delete `/var/lib/rancher/k3s/server`.

```bash
sudo systemctl stop k3s
sudo /usr/local/bin/k3s-killall.sh

TS=$(date +%Y%m%d_%H%M%S)
sudo mv /var/lib/rancher/k3s/agent/containerd /var/lib/rancher/k3s/agent/containerd.bak.$TS
sudo mkdir -p /var/lib/rancher/k3s/agent/containerd
sudo chown -R root:root /var/lib/rancher/k3s/agent/containerd

sudo systemctl start k3s
sleep 20
```

### Important shell-expansion gotcha

If you remove backups with `sudo rm -rf /path/*`, wildcard may expand before `sudo`.

Use root-side expansion:

```bash
sudo bash -lc 'rm -rf /var/lib/rancher/k3s/agent/containerd.bak.*'
```

Verify:

```bash
sudo ls -lah /var/lib/rancher/k3s/agent | grep containerd
df -h
kubectl describe node dli-control-plane | egrep -i "Taints|DiskPressure"
```

---

## 5) Optional for Small Root Disks: Lower ext4 Reserved Blocks

On small Raspberry Pi root partitions, ext4 reserved blocks can hide usable space.

Check current reserve:

```bash
sudo tune2fs -l /dev/mmcblk0p2 | egrep "Reserved block count|Block size"
```

Set reserve to 1% (recommended on root FS; avoid 0% on root):

```bash
sudo tune2fs -m 1 /dev/mmcblk0p2
df -h
```

---

## 6) Recover Workloads

After `DiskPressure=False`:

```bash
kubectl apply -f ~/dli/k8s/configmap.yaml
kubectl apply -f ~/dli/k8s/inference-stage-1-service.yaml
kubectl apply -f ~/dli/k8s/inference-stage-2-service.yaml
kubectl apply -f ~/dli/k8s/inference-stage-3-service.yaml
kubectl apply -f ~/dli/k8s/inference-stage-4-service.yaml
kubectl apply -f ~/dli/k8s/inference-gateway-service.yaml
kubectl apply -f ~/dli/k8s/inference-stage-1-deployment.yaml
kubectl apply -f ~/dli/k8s/inference-stage-2-deployment.yaml
kubectl apply -f ~/dli/k8s/inference-stage-3-deployment.yaml
kubectl apply -f ~/dli/k8s/inference-stage-4-deployment.yaml
kubectl apply -f ~/dli/k8s/inference-gateway-deployment.yaml
kubectl -n inference scale deploy --all --replicas=1
kubectl -n inference get pods -o wide -w
```

If any stage is stuck in `Init:CrashLoopBackOff`, inspect init logs:

```bash
kubectl -n inference logs <stage-pod-name> -c fetch-stage-partition --previous --tail=120
```

---

## 7) Sanity Checklist

- Node has no taint:
  - `kubectl describe node dli-control-plane | grep -i Taints`
- `DiskPressure=False`
- Free disk is comfortably above kubelet eviction threshold
- Deployments use current image tags (not stale old tags)
- `hf-hub` secret contains a valid read token

