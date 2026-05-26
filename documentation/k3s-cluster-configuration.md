# K3s Cluster Configuration

This document records the post-install cluster configuration commands for the distributed inference K3s cluster.

The cluster nodes are:

| Kubernetes Node Name | Physical Node | Role Label | IP |
|---|---|---|---|
| `dli-control-plane` | `dli-control-plane-host` | `role=master` | `<CONTROL_PLANE_IP>` |
| `dli-worker-1` | `dli-worker-1` | `role=client` | `<WORKER_1_IP>` |
| `dli-worker-2` | `dli-worker-2` | `role=client` | `<WORKER_2_IP>` |
| `dli-worker-3` | `dli-worker-3` | `role=client` | `<WORKER_3_IP>` |
| `dli-worker-4` | `dli-worker-4` | `role=client` | `<WORKER_4_IP>` |

## 1. Verify Cluster Nodes

Run on the master:

```bash
kubectl get nodes -o wide
```

Expected result:

```text
NAME         STATUS   ROLES                INTERNAL-IP
dli-control-plane   Ready    control-plane,etcd   <CONTROL_PLANE_IP>
dli-worker-1      Ready    <none>               <WORKER_1_IP>
dli-worker-2      Ready    <none>               <WORKER_2_IP>
dli-worker-3      Ready    <none>               <WORKER_3_IP>
dli-worker-4     Ready    <none>               <WORKER_4_IP>
```

## 2. Label Worker Nodes

Label all worker nodes as inference clients:

```bash
kubectl label node dli-worker-1 role=client
kubectl label node dli-worker-2 role=client
kubectl label node dli-worker-3 role=client
kubectl label node dli-worker-4 role=client
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

The Kubernetes node name for the master is `dli-control-plane`, not `dli-control-plane-host`.

This command is wrong:

```bash
kubectl label node dli-control-plane-host role=master
```

It fails because no Kubernetes node named `dli-control-plane-host` exists:

```text
Error from server (NotFound): nodes "dli-control-plane-host" not found
```

Use this instead:

```bash
kubectl label node dli-control-plane role=master
```

Explanation:

- `dli-control-plane-host` is the physical device name used in the documentation.
- `dli-control-plane` is the Kubernetes node name configured during K3s installation.
- Kubernetes commands must use Kubernetes node names.

## 4. Verify Labels

Run:

```bash
kubectl get nodes --show-labels
```

Expected labels:

```text
dli-control-plane   ... role=master
dli-worker-1      ... role=client
dli-worker-2      ... role=client
dli-worker-3      ... role=client
dli-worker-4     ... role=client
```

Example observed output:

```text
NAME         STATUS   ROLES                AGE     VERSION        LABELS
dli-control-plane   Ready    control-plane,etcd   88m     v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/arch=arm64,kubernetes.io/hostname=dli-control-plane,kubernetes.io/os=linux,node-role.kubernetes.io/control-plane=true,node-role.kubernetes.io/etcd=true,node.kubernetes.io/instance-type=k3s,role=master
dli-worker-4     Ready    <none>               29m     v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/arch=arm64,kubernetes.io/hostname=dli-worker-4,kubernetes.io/os=linux,node.kubernetes.io/instance-type=k3s,role=client
dli-worker-1      Ready    <none>               43m     v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/arch=arm64,kubernetes.io/hostname=dli-worker-1,kubernetes.io/os=linux,node.kubernetes.io/instance-type=k3s,role=client
dli-worker-2      Ready    <none>               9m41s   v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/arch=arm64,kubernetes.io/hostname=dli-worker-2,kubernetes.io/os=linux,node.kubernetes.io/instance-type=k3s,role=client
dli-worker-3      Ready    <none>               15m     v1.35.4+k3s1   beta.kubernetes.io/arch=arm64,beta.kubernetes.io/instance-type=k3s,beta.kubernetes.io/os=linux,kubernetes.io/hostname=dli-worker-3,kubernetes.io/os=linux,node.kubernetes.io/instance-type=k3s,role=client
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
dli-control-plane   role=master
dli-worker-1      role=client
dli-worker-2      role=client
dli-worker-3      role=client
dli-worker-4     role=client
namespace    inference
metrics      metrics-server
storage      local-path
```
