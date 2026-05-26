# Probleme K3s Intalnite Si Rezolvari

Acest document descrie problemele aparute in timpul instalarii clusterului K3s pentru proiectul de distributed inference si modul in care au fost rezolvate.

Cluster final:

| Nod | Rol | IP |
|---|---|---|
| `dli-control-plane` / `dli-control-plane-host` | control-plane + etcd | `<CONTROL_PLANE_IP>` |
| `dli-worker-1` | worker | `<WORKER_1_IP>` |
| `dli-worker-2` | worker | `<WORKER_2_IP>` |
| `dli-worker-3` | worker | `<WORKER_3_IP>` |
| `dli-worker-4` | worker | `<WORKER_4_IP>` |

## 1. `k3s.service` Nu Exista Dupa Instalare

Simptom:

```text
Unit k3s.service could not be found.
```

Comanda initiala:

```bash
curl -sfL https://get.k3s.io | sudo env INSTALL_K3S_EXEC="server ..." sh -
```

Problema:

- Scriptul K3s nu a fost descarcat.
- Optiunea `-s` din `curl -sfL` a ascuns eroarea reala.
- `sh -` a primit input gol, deci instalarea nu a rulat.

Diagnostic corect:

```bash
curl -fL https://get.k3s.io -o /tmp/install-k3s.sh
echo $?
ls -lh /tmp/install-k3s.sh
```

Rezolvare:

- Scriptul a fost descarcat separat in `/tmp/install-k3s.sh`.
- Instalarea a fost rulata numai dupa ce fisierul exista.

## 2. Eroare SSL: Certificatul Nu Este Inca Valid

Simptom:

```text
curl: (60) SSL certificate problem: certificate is not yet valid
```

Diagnostic:

```bash
date
timedatectl
curl -vI https://get.k3s.io
```

Problema:

- Ceasul sistemului era gresit.
- Nodul era setat pe o data din trecut.
- Din cauza datei gresite, certificatul TLS pentru `get.k3s.io` parea sa fie "din viitor".

Rezolvare:

```bash
sudo apt update
sudo apt install -y systemd-timesyncd
sudo systemctl enable --now systemd-timesyncd
sudo timedatectl set-ntp true
```

Verificare:

```bash
date
timedatectl
```

Rezultatul asteptat:

```text
System clock synchronized: yes
NTP service: active
```

## 3. IP Gresit Pentru Master

Simptom:

```text
listen tcp <WRONG_NODE_IP>:2380: bind: cannot assign requested address
```

Problema:

- Masterul a fost instalat initial cu:

```text
--node-ip <WRONG_NODE_IP>
--advertise-address <WRONG_NODE_IP>
```

- Dar IP-ul real al interfetei `wlan0` era:

```text
<CONTROL_PLANE_IP>/24
```

Diagnostic:

```bash
ip addr show wlan0
```

Rezolvare:

Instalarea a fost refacuta cu IP-ul corect:

```bash
sudo env INSTALL_K3S_EXEC="server --node-name dli-control-plane --write-kubeconfig-mode 644 --cluster-init --disable traefik --disable servicelb --flannel-backend vxlan --node-ip <CONTROL_PLANE_IP> --advertise-address <CONTROL_PLANE_IP>" sh /tmp/install-k3s.sh
```

Verificare:

```bash
sudo k3s kubectl get nodes -o wide
```

## 4. Client Instalate Fara Token

Simptom pe worker:

```text
Error: --token is required
```

Problema:

- Variabilele de mediu au fost rulate separat:

```bash
sudo env K3S_URL="https://<CONTROL_PLANE_IP>:6443"
K3S_TOKEN="..."
INSTALL_K3S_EXEC="agent ..." sh /tmp/install-k3s.sh
```

- Prima comanda doar a afisat environment-ul pentru root si a iesit.
- `K3S_TOKEN` nu a fost transmis catre installer si nici catre serviciul systemd.

Rezolvare:

Toate variabilele trebuie transmise in aceeasi comanda:

```bash
sudo env K3S_URL="https://<CONTROL_PLANE_IP>:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name dli-worker-1 --node-ip <WORKER_1_IP>" \
sh /tmp/install-k3s.sh
```

Pentru un worker instalat gresit:

```bash
sudo systemctl stop k3s-agent
sudo /usr/local/bin/k3s-agent-uninstall.sh
```

Apoi se reinstaleaza cu variabilele corecte.

## 5. Master Instabil: API Server Refused / ServiceUnavailable

Simptome:

```text
The connection to the server 127.0.0.1:6443 was refused
```

```text
Error from server (ServiceUnavailable): the server is currently unable to handle the request
```

Problema:

- `k3s.service` pornea, raspundea scurt, apoi iesea.
- Workerii incercau sa se conecteze in timp ce masterul repornea.
- Problema reala era in Flannel/VXLAN, nu in token sau in Kubernetes API.

Diagnostic:

```bash
sudo systemctl status k3s --no-pager -l
sudo journalctl -u k3s -b --no-pager | grep -Ei "fatal|shutdown|failed|flannel|address already|exited|panic|error" | tail -120
```

Eroarea importanta:

```text
flannel exited: failed to register flannel network:
failed to configure interface flannel.1:
failed to set interface flannel.1 to UP state: address already in use
```

## 6. Conflict Intre Cilium VXLAN Si Flannel VXLAN

Diagnostic:

```bash
ip -d link show type vxlan
sudo ss -lunp | grep -E ':8472|:51820'
```

Rezultat problematic:

```text
cilium_vxlan ... dstport 8472
flannel.1 ... dstport 8472
```

Problema:

- Exista un device VXLAN ramas de la Cilium: `cilium_vxlan`.
- K3s era configurat sa foloseasca Flannel cu VXLAN:

```text
--flannel-backend vxlan
```

- Flannel foloseste UDP port `8472`.
- `cilium_vxlan` ocupa acelasi port, deci Flannel nu putea aduce `flannel.1` in starea `UP`.

Rezolvare pe master:

```bash
sudo systemctl stop k3s
sudo /usr/local/bin/k3s-killall.sh

sudo ip link delete cilium_vxlan 2>/dev/null || true
sudo ip link delete flannel.1 2>/dev/null || true
sudo ip link delete cni0 2>/dev/null || true

sudo rm -rf /run/flannel
sudo rm -rf /var/lib/cni
```

S-a verificat ca nu exista manifest Cilium auto-aplicat de K3s:

```bash
sudo find /var/lib/rancher/k3s/server/manifests -iname '*cilium*' -o -iname '*cni*'
```

Pornire master:

```bash
sudo systemctl start k3s
sleep 60
sudo systemctl is-active k3s
sudo k3s kubectl get nodes -o wide
```

Verificare VXLAN corecta:

```bash
ip -d link show type vxlan
```

Rezultatul corect:

```text
flannel.1 ... dstport 8472
```

Nu trebuie sa mai existe:

```text
cilium_vxlan
```

## 7. `k3s-agent.service` Gasit Pe Master

Simptom:

```bash
ls -l /etc/systemd/system/k3s*.service
```

Rezultat:

```text
/etc/systemd/system/k3s-agent.service
/etc/systemd/system/k3s.service
```

Problema:

- Masterul trebuie sa ruleze `k3s.service`, nu `k3s-agent.service`.
- Un agent unit ramas pe master putea crea confuzie si stare locala gresita.

Rezolvare:

```bash
sudo systemctl stop k3s-agent 2>/dev/null || true
sudo systemctl disable k3s-agent 2>/dev/null || true
sudo rm -f /etc/systemd/system/k3s-agent.service /etc/systemd/system/k3s-agent.service.env
sudo systemctl daemon-reload
```

Verificare:

```bash
ls -l /etc/systemd/system/k3s*.service
systemctl list-units 'k3s*' --all --no-pager
```

Rezultat corect pe master:

```text
/etc/systemd/system/k3s.service
```

## 8. Workeri `NotReady` Dupa Instabilitatea Masterului

Simptome:

```text
dli-worker-4 NotReady
dli-worker-2  NotReady
```

Loguri intalnite pe agenti:

```text
server is not ready
connect: connection refused
apiserver not ready
NetworkPluginNotReady
```

Problema:

- Workerii au incercat sa se inregistreze in timp ce masterul era instabil.
- Unele servicii agent au ramas cu stare locala incompleta.

Rezolvare generala pentru un worker afectat:

```bash
sudo systemctl stop k3s-agent
sudo /usr/local/bin/k3s-agent-uninstall.sh
sudo /usr/local/bin/k3s-killall.sh 2>/dev/null || true

sudo rm -rf /run/flannel /var/lib/cni /var/lib/kubelet /var/lib/rancher/k3s
sudo ip link delete flannel.1 2>/dev/null || true
sudo ip link delete cni0 2>/dev/null || true
sudo ip link delete cilium_vxlan 2>/dev/null || true
sudo systemctl daemon-reload
```

Pe master, stergere nod vechi:

```bash
sudo k3s kubectl delete node NODE_NAME --ignore-not-found
```

Reinstalare worker:

```bash
sudo env K3S_URL="https://<CONTROL_PLANE_IP>:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name NODE_NAME --node-ip NODE_IP" \
sh /tmp/install-k3s.sh
```

## 9. `dli-worker-2` A Iesit Cu `Result: protocol`

Simptom:

```text
Job for k3s-agent.service failed because the service did not take the steps required by its unit configuration.
Result: protocol
ExecStart=/usr/local/bin/k3s agent ... exited, status=0/SUCCESS
```

Problema:

- Serviciul `k3s-agent` iesea imediat, desi systemd astepta un proces long-running.
- Cea mai rapida rezolvare a fost curatarea starii locale si reinstalarea agentului.

Rezolvare:

```bash
sudo systemctl stop k3s-agent
sudo /usr/local/bin/k3s-agent-uninstall.sh
sudo /usr/local/bin/k3s-killall.sh 2>/dev/null || true

sudo rm -rf /run/flannel /var/lib/cni /var/lib/kubelet /var/lib/rancher/k3s
sudo ip link delete flannel.1 2>/dev/null || true
sudo ip link delete cni0 2>/dev/null || true
sudo ip link delete cilium_vxlan 2>/dev/null || true
sudo systemctl daemon-reload
```

Pe master:

```bash
sudo k3s kubectl delete node dli-worker-2 --ignore-not-found
```

Reinstalare `dli-worker-2`:

```bash
sudo env K3S_URL="https://<CONTROL_PLANE_IP>:6443" \
K3S_TOKEN="<K3S_NODE_TOKEN>" \
INSTALL_K3S_EXEC="agent --node-name dli-worker-2 --node-ip <WORKER_2_IP>" \
sh /tmp/install-k3s.sh
```

## 10. Starea Finala

Verificare finala pe master:

```bash
sudo k3s kubectl get nodes -o wide
```

Rezultat final:

```text
NAME         STATUS   ROLES                INTERNAL-IP
dli-control-plane   Ready    control-plane,etcd   <CONTROL_PLANE_IP>
dli-worker-1      Ready    <none>               <WORKER_1_IP>
dli-worker-2      Ready    <none>               <WORKER_2_IP>
dli-worker-3      Ready    <none>               <WORKER_3_IP>
dli-worker-4     Ready    <none>               <WORKER_4_IP>
```

## Lectii Importante

- Nu folosi `curl -s` cand diagnostichezi instalari; ascunde erori importante.
- Verifica mereu data sistemului inainte de instalari HTTPS.
- `--node-ip` si `--advertise-address` trebuie sa fie IP-uri reale ale nodului.
- Variabilele `K3S_URL`, `K3S_TOKEN` si `INSTALL_K3S_EXEC` trebuie transmise in aceeasi comanda `sudo env`.
- Nu amesteca Cilium VXLAN si Flannel VXLAN pe acelasi host fara configuratie explicita.
- Daca masterul flappeaza, opreste workerii si repara masterul intai.
- Pentru noduri worker in stare corupta, cel mai simplu fix este uninstall + curatare CNI + reinstall.
- Tokenul de join este secret si trebuie rotit daca a fost expus.

## Rotire Token Dupa Expunere

Tokenul de join a fost expus in timpul depanarii. Dupa ce toate nodurile sunt `Ready`, se recomanda rotirea lui pe master:

```bash
sudo k3s token rotate
sudo cat /var/lib/rancher/k3s/server/node-token
```

Nodurile deja conectate raman in cluster. Tokenul nou este necesar doar pentru join-uri viitoare.

