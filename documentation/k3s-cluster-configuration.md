# K3s Cluster Configuration

This document records the post-install cluster configuration commands for the distributed inference K3s cluster.

The cluster nodes are:

| Kubernetes Node Name | Physical Node | Role Label | IP |
|---|---|---|---|
| `k3s-master` | `pinode6` | `role=master` | `192.168.201.225` |
| `pinode7` | `pinode7` | `role=client` | `192.168.201.226` |
| `pinode8` | `pinode8` | `role=client` | `192.168.201.227` |
| `pinode9` | `pinode9` | `role=client` | `192.168.201.228` |
| `pinode10` | `pinode10` | `role=client` | `192.168.201.229` |

## 1. Verify Cluster Nodes

Run on the master:

```bash
kubectl get nodes -o wide
```

Expected result:

```text
NAME         STATUS   ROLES                INTERNAL-IP
k3s-master   Ready    control-plane,etcd   192.168.201.225
pinode7      Ready    <none>               192.168.201.226
pinode8      Ready    <none>               192.168.201.227
pinode9      Ready    <none>               192.168.201.228
pinode10     Ready    <none>               192.168.201.229
```

## 2. Label Worker Nodes

Label all worker nodes as inference clients:

```bash
kubectl label node pinode7 role=client
kubectl label node pinode8 role=client
kubectl label node pinode9 role=client
kubectl label node pinode10 role=client
```

Explanation:

- `role=client` is a custom label.
- It can be used by workloads with `nodeSelector`.
- This label does not change the `ROLES` column shown by `kubectl get nodes`.

Example workload selector:

```yaml
nodeSelector:
  role: client
```

## 3. Label Master Node

The Kubernetes node name for the master is `k3s-master`, not `pinode6`.

This command is wrong:

```bash
kubectl label node pinode6 role=master
```

It fails because no Kubernetes node named `pinode6` exists:

```text
Error from server (NotFound): nodes "pinode6" not found
```

Use this instead:

```bash
kubectl label node k3s-master role=master
```

Explanation:

- `pinode6` is the physical device name used in the documentation.
- `k3s-master` is the Kubernetes node name configured during K3s installation.
- Kubernetes commands must use Kubernetes node names.

## 4. Verify Labels

Run:

```bash
kubectl get nodes --show-labels
```

Expected labels:

```text
k3s-master   ... role=master
pinode7      ... role=client
pinode8      ... role=client
pinode9      ... role=client
pinode10     ... role=client
```

Example observed output:

```text
NAME         STATUS   ROLES                AGE     VERSION        LABELS
k3s-master   Ready    control-plane,etcd   88m     v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/arch=arm64,kubernetes.io/hostname=k3s-master,kubernetes.io/os=linux,node-role.kubernetes.io/control-plane=true,node-role.kubernetes.io/etcd=true,node.kubernetes.io/instance-type=k3s,role=master
pinode10     Ready    <none>               29m     v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/arch=arm64,kubernetes.io/hostname=pinode10,kubernetes.io/os=linux,node.kubernetes.io/instance-type=k3s,role=client
pinode7      Ready    <none>               43m     v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/arch=arm64,kubernetes.io/hostname=pinode7,kubernetes.io/os=linux,node.kubernetes.io/instance-type=k3s,role=client
pinode8      Ready    <none>               9m41s   v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/arch=arm64,kubernetes.io/hostname=pinode8,kubernetes.io/os=linux,node.kubernetes.io/instance-type=k3s,role=client
pinode9      Ready    <none>               15m     v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/hostname=pinode9,kubernetes.io/os=linux,node.kubernetes.io/instance-type=k3s,role=client
```

## 5. Create Inference Namespace

Create the namespace for distributed inference workloads:

```bash
kubectl create namespace inference
```

Expected result:

```text
namespace/inference created
```

Verify:

```bash
kubectl get namespaces
```

Or:

```bash
kubectl get namespace inference
```

Expected result:

```text
NAME        STATUS   AGE
inference   Active   ...
```

## 6. Idempotent Namespace Command

If the namespace may already exist, use:

```bash
kubectl create namespace inference --dry-run=client -o yaml | kubectl apply -f -
```

This avoids an error if `inference` already exists.

## 7. Configure Metrics Server

Apply the upstream Metrics Server manifest:

```bash
kubectl apply -f \
https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

Observed output included warnings like:

```text
Warning: resource serviceaccounts/metrics-server is missing the kubectl.kubernetes.io/last-applied-configuration annotation
serviceaccount/metrics-server configured
deployment.apps/metrics-server configured
apiservice.apiregistration.k8s.io/v1beta1.metrics.k8s.io configured
```

Explanation:

- These warnings are not fatal.
- They happen when resources already exist but were not originally created with `kubectl apply`.
- K3s normally includes Metrics Server by default unless it was disabled.
- `kubectl apply` patches the missing `last-applied-configuration` annotation automatically and configures the existing resources.

Verify Metrics Server:

```bash
kubectl get deployment metrics-server -n kube-system
kubectl get apiservice v1beta1.metrics.k8s.io
kubectl top nodes
```

If `kubectl top nodes` does not work immediately, wait for the Metrics Server pod to become ready:

```bash
kubectl get pods -n kube-system -o wide | grep metrics-server
```

## 8. Configure Local Path Provisioner

Apply the upstream Rancher Local Path Provisioner manifest:

```bash
kubectl apply -f \
https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
```

Observed output:

```text
namespace/local-path-storage created
serviceaccount/local-path-provisioner-service-account created
role.rbac.authorization.k8s.io/local-path-provisioner-role created
deployment.apps/local-path-provisioner created
storageclass.storage.k8s.io/local-path configured
configmap/local-path-config created
```

Some resources may show warnings like:

```text
Warning: resource storageclasses/local-path is missing the kubectl.kubernetes.io/last-applied-configuration annotation
```

Explanation:

- The warnings are not fatal.
- K3s normally ships with a `local-path` storage class and local-path provisioner.
- Applying the upstream manifest can configure existing cluster-wide resources such as `storageclass/local-path`.
- It may also create a separate `local-path-storage` namespace and deployment.

Verify storage:

```bash
kubectl get storageclass
kubectl get pods -A -o wide | grep local-path
kubectl get namespace local-path-storage
```

Expected storage class:

```text
local-path
```

If more than one local-path provisioner is running, inspect both before deleting anything:

```bash
kubectl get deployments -A | grep local-path
kubectl get pods -A -o wide | grep local-path
```

## 9. Useful Configuration Checks

Show labels:

```bash
kubectl get nodes --show-labels
```

Show only custom role labels:

```bash
kubectl get nodes -L role
```

Show nodes with IPs:

```bash
kubectl get nodes -o wide
```

Show namespaces:

```bash
kubectl get ns
```

Show all pods in the cluster:

```bash
kubectl get pods -A -o wide
```

Show Metrics Server API:

```bash
kubectl get apiservice v1beta1.metrics.k8s.io
```

Show node metrics:

```bash
kubectl top nodes
```

Show storage classes:

```bash
kubectl get storageclass
```

## 10. Current Configuration Summary

The cluster configuration after labeling, namespace creation, Metrics Server configuration, and Local Path Provisioner configuration is:

```text
k3s-master   role=master
pinode7      role=client
pinode8      role=client
pinode9      role=client
pinode10     role=client
namespace    inference
metrics      metrics-server
storage      local-path
```
